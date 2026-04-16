import os
import csv
import glob
import numpy as np
import nibabel as nib
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from scipy.ndimage import sobel, zoom
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils.utils import seed_torch, load_checkpoint
from utils.config import config
from model.aae import AAE
from model.unet import UNet
from model.da_net import DA_NET3D
from scheduler.linear_scheduler import Diffusion
from dataset.adni_dataset import TwoDataset, crop

# -------------------------------------------------------------------------
# torchrun --nproc_per_node=<num_gpus> test_dapf_ldm.py
# -------------------------------------------------------------------------

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def build_and_load_models(device):

    # AAE (Decoder)
    aae = AAE().to(device)
    opt_dummy = torch.optim.Adam(aae.parameters(), lr=1e-8) # Dummy optimizer
    load_checkpoint(config.CHECKPOINT_AAE, aae, opt_dummy, 0, device)
    aae.eval()

    # DA-Net (Guidance)
    da_model = DA_NET3D().to(device)
    load_checkpoint(config.CHECKPOINT_DA, da_model, opt_dummy, 0, device)
    da_model.eval()

    # UNet (Denoiser) - Load the best checkpoint (EMA version usually)
    unet = UNet().to(device)
    if os.path.exists(config.CHECKPOINT_Unet):
        print(f"Loading UNet checkpoint from {config.CHECKPOINT_Unet}")
        load_checkpoint(config.CHECKPOINT_Unet, unet, opt_dummy, 0, device)
    else:
        raise FileNotFoundError(f"UNet checkpoint not found: {config.CHECKPOINT_Unet}")
    unet.eval()

    diffusion = Diffusion()
    return aae, unet, da_model, diffusion

def load_selected_masks(dir_path, selected_ids=[2,3,9,10,13,14,19,20]):
    """
    Load specific ROI masks for evaluation.
    Ids correspond to AD-critical regions (e.g., Precuneus, Hippocampus).
    """
    mask_files = sorted(glob.glob(os.path.join(dir_path, "*.nii.gz")))
    masks_list = []
    for idx in selected_ids:
        file = mask_files[idx]
        data = nib.load(file).get_fdata().astype(np.float32)
        mid = crop(data) # Crop to match model output size
        masks_list.append(mid)
    return np.stack(masks_list) # [N_ROI, D, H, W]

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance between two multivariate Gaussians.
    Used for FMD (Frechet Modality Distance) calculation.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    # Product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def run_test(aae, unet, da_model, diffusion, device, rank):
    """
    Main inference 
    """
    # 1. Dataset
    test_ds = TwoDataset(root_MRI=config.test_FDG_MRI, root_FDG=config.test_FDG, stage="test", csv_path=config.test_FDG_CSV)
    test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=1, sampler=test_sampler, num_workers=config.numworker, pin_memory=True)
    
    # 2. ROI Masks
    masks = load_selected_masks(config.atlas_dir)
    # Union of all ROIs for Structural Consistency calculation
    union_mask = np.any(masks > 0, axis=0).astype(np.float32) 
    roi_bool_idx = union_mask > 0.5

    # 3. Metrics Storage
    local_metrics = {'psnr': 0.0, 'ssim': 0.0, 'mae': 0.0, 'mae_roi': 0.0, 'roi_psnr': 0.0}
    n_samples = 0
    
    # Storage for FMD calculation (Features from AAE Encoder)
    features_real = []
    features_fake = []
    
    per_sample_results = []

    with torch.no_grad():
        for MRI, FDG, sample_id, label, age in tqdm(test_loader, desc="Inference", disable=(rank!=0)):
            # Prepare Input
            label = label.to(device)
            ages = age.to(device)
            MRI_tensor = MRI.unsqueeze(1).to(device)
            
            # Predict Guidance (Saliency & Pseudo-label)
            class_scores, saliency = da_model(MRI_tensor)
            pseudo_label = class_scores.argmax(dim=1)
            
            # Diffusion Sampling (DDIM)
            # Use pseudo_label for fully automated inference
            sampled_latent = diffusion.sample_ddim(unet, MRI_tensor, label=pseudo_label, ages=ages, saliency=saliency)
            
            # Decode to Pixel Space
            syn_FDG = aae.decoder(sampled_latent)
            
            # Post-processing
            syn_FDG_np = torch.clamp(syn_FDG, 0, 1).cpu().numpy().squeeze()
            # Re-normalize to 0-1 for metric calc
            syn_FDG_np = (syn_FDG_np - syn_FDG_np.min()) / (syn_FDG_np.max() - syn_FDG_np.min() + 1e-8)
            
            FDG_np = FDG.numpy().squeeze()
            FDG_np = (FDG_np - FDG_np.min()) / (FDG_np.max() - FDG_np.min() + 1e-8)
            
            # --- Metrics Calculation ---
            cur_psnr = psnr(FDG_np, syn_FDG_np, data_range=1.0)
            cur_ssim = ssim(FDG_np, syn_FDG_np, data_range=1.0)
            cur_mae = np.mean(np.abs(syn_FDG_np - FDG_np))
            
            cur_mae_roi = np.mean(np.abs(syn_FDG_np[roi_bool_idx] - FDG_np[roi_bool_idx]))
            
            mse_roi = np.mean((syn_FDG_np[roi_bool_idx] - FDG_np[roi_bool_idx])**2)
            cur_roi_psnr = 10 * np.log10(1.0 / (mse_roi + 1e-12))
            
            local_metrics['psnr'] += cur_psnr
            local_metrics['ssim'] += cur_ssim
            local_metrics['mae'] += cur_mae
            local_metrics['mae_roi'] += cur_mae_roi
            local_metrics['roi_psnr'] += cur_roi_psnr
            n_samples += 1
            
            # Save Sample Result
            if isinstance(sample_id, list): sample_id = sample_id[0]
            per_sample_results.append({
                'id': str(sample_id),
                'psnr': cur_psnr, 'ssim': cur_ssim, 
                'mae_roi': cur_mae_roi, 'roi_psnr': cur_roi_psnr
            })
            
            real_tensor = torch.from_numpy(FDG_np).float().unsqueeze(0).unsqueeze(0).to(device)
            fake_tensor = torch.from_numpy(syn_FDG_np).float().unsqueeze(0).unsqueeze(0).to(device)
            
            z_real = aae.encoder(real_tensor)
            z_fake = aae.encoder(fake_tensor)
            
            z_real = F.adaptive_avg_pool3d(z_real, 1).view(-1).cpu().numpy()
            z_fake = F.adaptive_avg_pool3d(z_fake, 1).view(-1).cpu().numpy()
            
            features_real.append(z_real)
            features_fake.append(z_fake)
            
            if rank == 0:
                save_dir = os.path.join(config.exp_root, config.exp_ldm, "generated_samples")
                os.makedirs(save_dir, exist_ok=True)
                ref_img = nib.load(config.path)
                nifti_img = nib.Nifti1Image(syn_FDG_np, ref_img.affine, ref_img.header)
                nib.save(nifti_img, os.path.join(save_dir, f"{sample_id}_syn.nii.gz"))

    metrics_tensor = torch.tensor([
        local_metrics['psnr'], local_metrics['ssim'], local_metrics['mae'],
        local_metrics['mae_roi'], local_metrics['roi_psnr'],
        float(n_samples)
    ], device=device)
    
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    
    total_samples = int(metrics_tensor[5].item())
    avg_psnr = metrics_tensor[0].item() / total_samples
    avg_ssim = metrics_tensor[1].item() / total_samples
    avg_mae = metrics_tensor[2].item() / total_samples
    avg_mae_roi = metrics_tensor[3].item() / total_samples
    avg_roi_psnr = metrics_tensor[4].item() / total_samples
    
    # --- FMD Calculation 
    if rank == 0:
        feat_real = np.stack(features_real)
        feat_fake = np.stack(features_fake)
        
        mu_real, sigma_real = np.mean(feat_real, axis=0), np.cov(feat_real, rowvar=False)
        mu_fake, sigma_fake = np.mean(feat_fake, axis=0), np.cov(feat_fake, rowvar=False)
        
        fmd_score = frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        
        csv_path = os.path.join(config.exp_root, config.exp_ldm, "test_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["ID", "PSNR", "SSIM", "MAE_ROI", "ROI_PSNR"])
            for res in per_sample_results:
                w.writerow([res['id'], f"{res['psnr']:.4f}", f"{res['ssim']:.4f}", 
                            f"{res['mae_roi']:.4f}", f"{res['roi_psnr']:.4f}"])
            w.writerow([])
            w.writerow(["AVERAGE", f"{avg_psnr:.4f}", f"{avg_ssim:.4f}", 
                        f"{avg_mae_roi:.4f}", f"{avg_roi_psnr:.4f}"])
            w.writerow(["FMD", f"{fmd_score:.4f}"])
            
        print(f"\n[Test Finished] Avg PSNR: {avg_psnr:.4f} | Avg SSIM: {avg_ssim:.4f}")
        print(f"Lesion Metrics - ROI-PSNR: {avg_roi_psnr:.4f} | ROI-MAE: {avg_mae_roi:.4f}")
        print(f"Distribution Metric - FMD: {fmd_score:.4f}")

def main():
    seed_torch()
    local_rank, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    try:
        aae, unet, da, diff = build_and_load_models(device)
        run_test(aae, unet, da, diff, device, rank)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
