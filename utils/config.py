import torch
import os
from typing import Dict

class ProjectConfig:
    """
    Global Configuration for DAPF-LDM.
    
    Paper: DAPF-LDM: Diagnosis-Aware Latent Diffusion for Pathology-Focused MRI-to-PET Synthesis
    """
    # -------------------------------------------------------------------------
    # Hardware & System / 硬件与系统
    # -------------------------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    numworker: int = 16       # DataLoader workers
    seed: int = 42            # Random seed
    
    # -------------------------------------------------------------------------
    # DAPF Block Hyperparameters 
    # (Diagnosis-Aware Pathology-Focused Block)
    # -------------------------------------------------------------------------
    atlas_dir: str = "./datas/atlas_masks/"  # Path to anatomical ROI masks (22 regions)
    num_regions: int = 22     # Harvard-Oxford atlas selected regions
    in_channels: int = 16     # Channels entering the DAPF block (from Cond-Net)
    
    # Anatomical Prior Weights (pi_k) initialized based on literature 
    prior_weights: Dict[int, float] = {
        0: 0.0, 1: 0.3, 2: 1.0, 3: 0.3, 4: 0.8, 5: 0.3, 6: 0.3,
        7: 0.3, 8: 0.3, 9: 1.0, 10: 1.0, 11: 0.8, 
        12: 0.3, 13: 1.0, 14: 0.3, 15: 0.8, 16: 0.3, 17: 0.3,
        18: 0.3, 19: 1.0, 20: 1.0, 21: 0.8
    }
    
    # Adaptive Weighting Parameters
    lambda_param: float = 0.6   # Balancing factor (lambda) between Prior and MMD
    gamma: float = 2.0          # Upper bound for spatial weights
    rbf_sigma: float = 1.0      # Sigma for Gaussian RBF kernel in MMD
    alpha: float = 0.2          # Age modulation scaling factor
    use_multi_kernel: bool = False # Whether to use multi-kernel MMD
    fusion_gamma: float = 0.1   # Learnable scalar init for Residual Gated Modulation
    
    # -------------------------------------------------------------------------
    # Training Hyperparameters 
    # -------------------------------------------------------------------------
    phase: str = "train"        # 'encoding', 'train', 'test'
    noiseSteps: int = 1000      # T steps
    latent_dim: int = 1         # Latent space channels
    learning_rate: float = 1e-4 # LR for AAE/DA-Net/ResCNN
    batch_size: int = 6         # Batch size (constrained by 3D volume memory)
    epochs: int = 200           # Training epochs
    time_dim: int = 128         # Time embedding dimension
    num_classes: int = 3        # CN, MCI, AD
    
    # Loss Weights
    Lambda: float = 100.0       # AAE L1 recon weight
    lambda_ortho: float = 0.01  # Orthogonal decomposition loss weight
    lambda_reg: float = 0.1     # DAPF regularization weight (MMD + Alignment)
    lambda_fusion: float = 0.5  # Cross-Modal Fusion-Perceptual loss weight

    # Conditioning Strategies
    use_saliency: bool = True   # Enable Saliency Map from DA-Net
    cfg_drop_prob: float = 0.1  # Probability for Pseudo-label injection (Robust Conditioning)

    # -------------------------------------------------------------------------
    # Directories & Paths 
    # -------------------------------------------------------------------------
    exp_root: str = "result"
    
    # Experiment IDs
    exp_aae: str = "exp_1/"         # Stage I: AAE
    exp_da: str = "exp_da/"         # Stage I: DA-Net
    exp_mri_pet: str = "exp_mri_pet/" # Stage I: ResCNN (Perceptual Loss)
    exp_ldm: str = "exp_2/"         # Stage II: LDM (Main)
    
    # Dataset Paths
    # 1. Raw Data
    train_FDG: str = "./datas/FDG/PET/train/"
    val_FDG: str = "./datas/FDG/PET/val/"
    test_FDG: str = "./datas/FDG/PET/test/"
    
    train_FDG_MRI: str = "./datas/FDG/MRI/train/"
    val_FDG_MRI: str = "./datas/FDG/MRI/val/"
    test_FDG_MRI: str = "./datas/FDG/MRI/test/"
    
    # 2. Latent Data (Encoded by AAE)
    latent_FDG: str = "./datas/FDG/PET/latent_FDG/"
    
    # 3. CSV Info
    train_FDG_CSV: str = "./datas/FDG/MRI/FDG_train_MRI_info.csv"
    val_FDG_CSV: str = "./datas/FDG/MRI/FDG_val_MRI_info.csv"
    test_FDG_CSV: str = "./datas/FDG/MRI/FDG_test_MRI_info.csv"
    
    # 4. Checkpoints (Absolute paths constructed dynamically)
    CHECKPOINT_AAE: str = os.path.join(exp_root, exp_aae, "AAE.pth.tar")
    CHECKPOINT_DISC: str = os.path.join(exp_root, exp_aae, "discriminator.pth.tar")
    CHECKPOINT_DA: str = os.path.join(exp_root, exp_da, "DA_NET.pth.tar")
    CHECKPOINT_ResCNN: str = os.path.join(exp_root, exp_mri_pet, "ResCNN.pth.tar")
    CHECKPOINT_Unet: str = os.path.join(exp_root, exp_ldm, "Unet.pth.tar")
    path: str = "./datas/FDG/PET/train/002_S_0295-2011-06-09-PET-0001.nii.gz" 
# Instantiate
config = ProjectConfig()
