import os
import random
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Tuple
from utils.utils import seed_torch, save_checkpoint, load_checkpoint, write_csv_header
from utils.config import config
from model.aae import AAE
from model.unet import UNet, EMA
from model.multi_modal_res_cnn_3d import MultiModalResCNN3D
from model.da_net import DA_NET3D  
from scheduler.linear_scheduler import Diffusion
from dataset.adni_dataset import TwoDataset

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# torchrun --nproc_per_node=<num_gpus> train_dapf_ldm.py
# -------------------------------------------------------------------------

def setup_distributed():
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank()

def cleanup():

    if dist.is_initialized():
        dist.destroy_process_group()

# -------------------------------------------------------------------------
# Helper: Perceptual Loss 
# -------------------------------------------------------------------------

def compute_rescnn_loss(mri: torch.Tensor, real_pet: torch.Tensor, 
                       gen_pet: torch.Tensor, perceptual_net: nn.Module) -> torch.Tensor:
    """
    Compute Cross-Modal Fusion-Perceptual Loss.
    
    Args:
        mri: Input MRI volume 
        real_pet: Ground truth PET 
        gen_pet: Synthesized PET 
        perceptual_net: Frozen MultiModalResCNN3D 
    """
    # Extract multi-scale features 
    # returns: [Scale1, Scale2, Scale3, Fused]
    with torch.no_grad():
        real_feats_list = perceptual_net.extract_features(mri, real_pet)
    
    gen_feats_list = perceptual_net.extract_features(mri, gen_pet)
    
    total_loss = 0.0
    # Weighted sum of feature losses (Higher weight for fusion layer)
    weights = [1.0, 1.0, 1.0, 2.0] 
    
    for i, (real, gen) in enumerate(zip(real_feats_list, gen_feats_list)):
        layer_loss = F.mse_loss(gen, real)
        total_loss += layer_loss * weights[i]
        
    return total_loss

# -------------------------------------------------------------------------
# Main Training 
# -------------------------------------------------------------------------

def train_LDM(rank: int, local_rank: int):
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        result_dir = os.path.join(config.exp_root, config.exp_ldm) 
        os.makedirs(result_dir, exist_ok=True)
        loss_path = os.path.join(result_dir, "loss_curve.csv")
        val_path  = os.path.join(result_dir, "validation.csv")
        
        write_csv_header(loss_path, ["Epoch", "MSE_loss", "Fusion_loss", "Reg_loss", "Total_loss"])
        write_csv_header(val_path,  ["Epoch", "PSNR", "SSIM"])

    
    # A. AAE 
    aae = AAE().to(device)
    temp_opt = optim.Adam(aae.parameters(), lr=1e-4) 
    load_checkpoint(config.CHECKPOINT_AAE, aae, temp_opt, config.learning_rate, device)
    aae.eval()
    for param in aae.parameters(): param.requires_grad = False

    # B. DA-Net 
    da_net = DA_NET3D().to(device)
    temp_opt_da = optim.Adam(da_net.parameters(), lr=1e-4)
    load_checkpoint(config.CHECKPOINT_DA, da_net, temp_opt_da, config.learning_rate, device)
    da_net.eval()
    for param in da_net.parameters(): param.requires_grad = False

    # C. Perceptual Network 
    perceptual_net = MultiModalResCNN3D(num_classes=3).to(device)
    if os.path.exists(config.CHECKPOINT_ResCNN):
        checkpoint = torch.load(config.CHECKPOINT_ResCNN, map_location=device)
        perceptual_net.load_state_dict(checkpoint['state_dict'])
        if rank == 0: print(f"Loaded ResCNN from {config.CHECKPOINT_ResCNN}")
    perceptual_net.eval()
    for param in perceptual_net.parameters(): param.requires_grad = False
    
    unet = UNet().to(device)
    unet = SyncBatchNorm.convert_sync_batchnorm(unet)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    opt_unet = optim.AdamW(unet.parameters(), 
                      lr=5e-5,              
                      betas=(0.9, 0.999),   
                      weight_decay=0.05,    
                      eps=1e-8)
    
    scheduler = CosineAnnealingLR(opt_unet, T_max=config.epochs, eta_min=5e-7)
    
    # EMA 
    ema = EMA(0.9999)
    ema_unet = copy.deepcopy(unet.module)
    ema_unet.eval()
    for param in ema_unet.parameters(): param.requires_grad_(False)   

    if os.path.exists(config.CHECKPOINT_Unet):
        if rank == 0: print(f"Resuming UNet from {config.CHECKPOINT_Unet}")
        load_checkpoint(config.CHECKPOINT_Unet, unet, opt_unet, 5e-5, device)

    criterion = nn.MSELoss()
    diffusion = Diffusion()
    best_score = -float('inf')
    epoch_threshold = 1  

    for epoch in range(1, config.epochs + 1):
        if rank == 0: print(f"DAPF-LDM Epoch {epoch}/{config.epochs}")
        
        unet.train()

        train_ds = TwoDataset(root_MRI=config.train_FDG_MRI,
                              root_FDG=config.latent_FDG,
                              stage="train",
                              csv_path=config.train_FDG_CSV)
        sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(train_ds,
                                  batch_size=config.batch_size,
                                  sampler=sampler,
                                  num_workers=config.numworker,
                                  pin_memory=True,
                                  drop_last=True)
        sampler.set_epoch(epoch)
        
        losses = {'mse': 0.0, 'fusion': 0.0, 'reg': 0.0, 'total': 0.0}
        
        for MRI, latent_FDG, _ , label, age in tqdm(train_loader, desc="Training", disable=(rank!=0)):
            label = label.to(device)
            ages = age.to(device)
            MRI = MRI.unsqueeze(1).to(device)       # [B, 1, D, H, W]
            latent_FDG = latent_FDG.unsqueeze(1).to(device) # [B, 1, D, H, W]
            
            # 1. Diagnosis-Aware Guidance Extraction (Frozen DA-Net)
            with torch.no_grad():
                class_scores, saliency = da_net(MRI)
                pseudo_label = class_scores.argmax(dim=1)
            
            # 2. Probabilistic Pseudo-Label Injection (p=0.2)
            if random.random() < config.cfg_drop_prob:
                label = pseudo_label    
            
            # 3. Diffusion Forward Process (Add Noise)
            t = diffusion.sample_timesteps(latent_FDG.shape[0]).to(device)      
            x_t, _ = diffusion.noise_images(latent_FDG, t)
            
            # 4. UNet Prediction (Signal Reconstruction: x0-prediction)
            predicted_x0, reg_loss, cond_fused = unet(
                x_t, MRI, t, label=label, ages=ages, saliency=saliency)
            
            # 5. Loss Calculation 
            
            # A. Reconstruction Loss 
            mse_loss = criterion(predicted_x0, latent_FDG)
            
            # B. Cross-Modal Fusion-Perceptual Loss 
            # Decode latents to pixel space for perceptual calculation
            with torch.set_grad_enabled(True): 
                decoded_syn_pet = aae.decoder(predicted_x0)
            with torch.no_grad():
                decoded_real_pet = aae.decoder(latent_FDG)
            
            perc_loss = compute_rescnn_loss(MRI, decoded_real_pet, decoded_syn_pet, perceptual_net)
            
            # C. Total Loss 
            # DAPF Regularization + Fusion + MSE
            loss = mse_loss + config.lambda_reg * reg_loss + config.lambda_fusion * perc_loss

            # 6. Optimization 
            opt_unet.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            opt_unet.step()
            
            ema.step_ema(ema_unet, unet.module)

            losses['mse'] += mse_loss.item()
            losses['reg'] += reg_loss.item()
            losses['fusion'] += perc_loss.item()
            losses['total'] += loss.item()

        if rank == 0:
            avg_losses = {k: v / len(train_loader) for k, v in losses.items()}
            with open(loss_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, 
                                        round(avg_losses['mse'], 6), 
                                        round(avg_losses['fusion'], 6), 
                                        round(avg_losses['reg'], 6), 
                                        round(avg_losses['total'], 6)])

        # Validation 
        if epoch % epoch_threshold == 0:
            unet.eval() 
            
            local_psnr_sum = 0.0
            local_ssim_sum = 0.0
            
            val_ds = TwoDataset(root_MRI=config.val_FDG_MRI,
                                root_FDG=config.val_FDG, 
                                stage="val",
                                csv_path=config.val_FDG_CSV)
            val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler,
                                    num_workers=config.numworker, pin_memory=True)
            
            with torch.no_grad():
                for MRI, FDG, _, label, age in tqdm(val_loader, desc="Validation", disable=(rank != 0)):
                    label = label.to(device)
                    ages = age.to(device)
                    MRI = MRI.unsqueeze(1).to(device)
                    
                    class_scores, saliency = da_net(MRI)
                    # For validation, we can use pseudo labels or ground truth. 
                    # Here using pseudo labels to simulate inference.
                    pseudo_label = class_scores.argmax(dim=1)
                    
                    sampled_latent = diffusion.sample_ddim(ema_unet, MRI, label=pseudo_label, ages=ages, saliency=saliency)
                    
                    syn_FDG = aae.decoder(sampled_latent)
                    syn_FDG = torch.clamp(syn_FDG, 0, 1)
                    syn_FDG = syn_FDG.detach().cpu().numpy().squeeze().astype(np.float32)
                    syn_FDG = (syn_FDG - syn_FDG.min()) / (syn_FDG.max() - syn_FDG.min() + 1e-8)              

                    FDG = FDG.detach().cpu().numpy().squeeze().astype(np.float32)
                    FDG = (FDG - FDG.min()) / (FDG.max() - FDG.min() + 1e-8)

                    local_psnr_sum += psnr(FDG, syn_FDG, data_range=1.0)
                    local_ssim_sum += ssim(FDG, syn_FDG, data_range=1.0)
            
            psnr_sum_tensor = torch.tensor(local_psnr_sum).to(device)
            ssim_sum_tensor = torch.tensor(local_ssim_sum).to(device)
            dist.all_reduce(psnr_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(ssim_sum_tensor, op=dist.ReduceOp.SUM)
            
            total_val = len(val_ds)
            avg_psnr = psnr_sum_tensor.item() / total_val
            avg_ssim = ssim_sum_tensor.item() / total_val
            
            if rank == 0:
                with open(val_path, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch, round(avg_psnr, 3), round(avg_ssim, 3)])
                
                print(f"Epoch {epoch} Val: PSNR={avg_psnr:.3f}, SSIM={avg_ssim:.3f}")

                # Save Best Model 
                score = avg_psnr + 10 * avg_ssim
                if score > best_score:
                    best_score = score
                    save_checkpoint(ema_unet, opt_unet, filename=config.CHECKPOINT_Unet)
                    print(f"Saved Best Model at Epoch {epoch}")

        scheduler.step()
    
    cleanup()

if __name__ == '__main__':
    seed_torch()
    local_rank, rank = setup_distributed()

    train_LDM(rank, local_rank)
