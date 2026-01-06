import os
import random  
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader,DistributedSampler
from torch import optim
from torch.nn import functional as F
from torch.cuda import amp
from tqdm import tqdm
from utils.utils import *
from model.aae import AAE, Discriminator
from dataset.ADNI_dataset import OneDataset
from utils.config import config
import csv
import warnings
import nibabel as nib

warnings.filterwarnings("ignore")

# torchrun --nproc_per_node=<num_gpus> train_aae.py


def setup_distributed():
    """
    Initialize distributed training (NCCL backend).
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank()

def cleanup():
    """
    Clean up distributed training environment.
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def train_AAE(rank, local_rank):

    device = torch.device(f"cuda:{local_rank}")
    # Prepare result directory and CSVs (only on rank 0)
    result_dir = os.path.join("result", config.exp)
    if rank == 0:
        os.makedirs(result_dir, exist_ok=True)
        loss_csv = os.path.join(result_dir, "loss_curve.csv")
        val_csv  = os.path.join(result_dir, "validation.csv")
        test_csv = os.path.join(result_dir, "test.csv")
        write_csv_header(loss_csv, ["Epoch", "recon_loss", "disc_loss"])
        write_csv_header(val_csv, ["Epoch", "PSNR", "SSIM"])
        write_csv_header(test_csv, ["Epoch", "PSNR", "SSIM"])
    else:
        loss_csv = val_csv = test_csv = None

    # Build models
    model = AAE().to(device)
    disc  = Discriminator().to(device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    disc  = SyncBatchNorm.convert_sync_batchnorm(disc)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    disc  = DDP(disc, device_ids=[local_rank], output_device=local_rank)

    optG  = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    optD  = optim.Adam(disc.parameters(),  lr=config.learning_rate, betas=(0.5, 0.999))
    scaler_G = amp.GradScaler()
    scaler_D = amp.GradScaler()

    # Load checkpoints if exist
    if config.CHECKPOINT_AAE and os.path.exists(config.CHECKPOINT_AAE):
        load_checkpoint(config.CHECKPOINT_AAE, model, optG, config.learning_rate, device)
    if config.CHECKPOINT_DISC and os.path.exists(config.CHECKPOINT_DISC):
        load_checkpoint(config.CHECKPOINT_DISC, disc, optD, config.learning_rate, device)

    best_score = -float('inf')
    epoch_threshold = 30  

    for epoch in range(1, config.epochs + 1):
        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch}/{config.epochs}")
        model.train()  
        disc.train()

        train_ds = OneDataset(root_FDG=config.train_FDG, stage="train")
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.numworker,
            pin_memory=True,
            drop_last=True
        )
        train_sampler.set_epoch(epoch)
        running_recon = 0.0
        running_disc  = 0.0

        for FDG, name in tqdm(loader, desc="Training", disable=(rank!=0)):
            # FDG = torch.tensor(np.expand_dims(FDG, 1), device=device)
            FDG = np.expand_dims(FDG, axis=1)
            FDG = torch.tensor(FDG)
            FDG = FDG.to(device)


            with amp.autocast():
                decoded = model(FDG)
                real_pred = disc(FDG)
                fake_pred = disc(decoded.detach())

                recon_loss = F.l1_loss(decoded, FDG)
                g_loss = -fake_pred.mean()
                lossG = recon_loss * config.Lambda + g_loss

                d_loss_real = F.relu(1. - real_pred).mean()
                d_loss_fake = F.relu(1. + fake_pred).mean()
                lossD = (d_loss_real + d_loss_fake) * 0.5


            optG.zero_grad()
            scaler_G.scale(lossG).backward(retain_graph=True)
            scaler_G.step(optG)
            scaler_G.update()

            optD.zero_grad()
            scaler_D.scale(lossD).backward()
            scaler_D.step(optD)
            scaler_D.update()

            running_recon += recon_loss.item()
            running_disc  += lossD.item()


        if rank == 0:
            with open(loss_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch,
                                          running_recon/len(loader),
                                          running_disc/len(loader)])

        if epoch > epoch_threshold and rank == 0:

            model.eval()
            psnr_sum = 0.0
            ssim_sum = 0.0
            val_ds = OneDataset(root_FDG=config.val_FDG, stage="val")
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                        num_workers=config.numworker, pin_memory=True, drop_last=True)
            with torch.no_grad():
                for FDG, name in tqdm(val_loader, desc="Validation"):
                    FDG = np.expand_dims(FDG, axis=1)
                    FDG = torch.tensor(FDG)
                    FDG = FDG.to(device)
               
                    decoded = model(FDG)
                    decoded = torch.clamp(decoded,0,1)
                    decoded = decoded.detach().cpu().numpy()
                    decoded = np.squeeze(decoded)
                    decoded = decoded.astype(np.float32)
                    
                    FDG = FDG.detach().cpu().numpy()
                    FDG = np.squeeze(FDG)
                    FDG = FDG.astype(np.float32)
                    
                    psnr_sum += psnr(FDG, decoded, data_range=1.0)
                    ssim_sum += ssim(FDG, decoded, data_range=1.0)
                    
            avg_psnr = psnr_sum / len(val_loader)
            avg_ssim = ssim_sum / len(val_loader)
            with open(val_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, round(avg_psnr,3), round(avg_ssim,3)])

            score = avg_psnr + 100 * avg_ssim
            if score > best_score:
                best_score = score
                save_checkpoint(model, optG, filename=config.CHECKPOINT_AAE)
                save_checkpoint(disc, optD, filename=config.CHECKPOINT_DISC)


                test_ds = OneDataset(root_FDG=config.test_FDG, stage="test")
                test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                        num_workers=config.numworker, pin_memory=True, drop_last=True)
                test_psnr = 0.0
                test_ssim = 0.0
                with torch.no_grad():
                    for FDG, name in tqdm(test_loader, desc="Testing"):
                        FDG = np.expand_dims(FDG, axis=1)
                        FDG = torch.tensor(FDG)
                        FDG = FDG.to(device)
                        
                        
                        decoded = model(FDG)
                        decoded = torch.clamp(decoded,0,1)
                        decoded = decoded.detach().cpu().numpy()
                        decoded = np.squeeze(decoded)
                        decoded = decoded.astype(np.float32)
                        
                        FDG = FDG.detach().cpu().numpy()
                        FDG = np.squeeze(FDG)
                        FDG = FDG.astype(np.float32)
                        
                        test_psnr += psnr(FDG, decoded, data_range=1.0)
                        test_ssim += ssim(FDG, decoded, data_range=1.0)
                with open(test_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch,
                                            round(test_psnr/len(test_loader),3),
                                            round(test_ssim/len(test_loader),3)])

def encoding(rank, local_rank):
    device = torch.device(f"cuda:{local_rank}")
    model = AAE().to(device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate, device)
    image = nib.load(config.path)

    dataset = OneDataset(root_FDG=config.train_FDG, stage = "NO")
    loader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.numworker,pin_memory=True)
    # loop = tqdm(loader, leave=True)
    loop = tqdm(loader, disable=(rank!= 0), leave=True, desc="Encoding")

    for idx, (FDG,name) in enumerate(loop):
        FDG = np.expand_dims(FDG, axis=1)
        FDG = torch.tensor(FDG)
        FDG = FDG.to(device)
        
        latent_FDG = model.encoder(FDG)
        latent_FDG = latent_FDG.detach().cpu().numpy()
        latent_FDG = np.squeeze(latent_FDG)
        latent_FDG = latent_FDG.astype(np.float32)

        latent_FDG = nib.Nifti1Image(latent_FDG, image.affine)
        nib.save(latent_FDG, config.latent_FDG+str(name[0]))

if __name__ == '__main__':
    seed_torch()
    local_rank, rank = setup_distributed()
    if config.phase == "train":
        train_AAE(rank, local_rank)
    elif config.phase == "encoding":
        encoding(rank, local_rank)

