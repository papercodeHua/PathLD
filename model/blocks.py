import math
import torch
from torch import nn
from utils.config import config
from inspect import isfunction
from typing import List, Dict, Tuple
import torch.nn.functional as F

# Basic components

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=16, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)

cimport math
import torch
from torch import nn
from utils.config import config
from typing import List, Dict, Tuple
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Diagnosis-Aware Pathology-Focused (DAPF) Block
# -------------------------------------------------------------------------

class DAPFBlock(nn.Module):
    def __init__(self, 
                 num_regions: int, 
                 in_channels: int,
                 prior_weights: Dict[int, float],
                 lambda_param: float = 0.7,
                 gamma: float = 2.0,
                 rbf_sigma: float = 1.0,
                 alpha: float = 0.1,
                 use_multi_kernel: bool = True,
                 fusion_gamma: float = 0.1):
        """      
        Args:
            num_regions: Number of anatomical ROIs 
            in_channels: Input feature channels 
            prior_weights: Dictionary of prior weights for each region 
            lambda_param: Balance parameter between prior and data-driven evidence 
            gamma: Upper bound for weights 
            rbf_sigma: Sigma for RBF kernel 
            alpha: Scaling factor for age modulation 
            use_multi_kernel: Whether to use multi-kernel MMD 
            fusion_gamma: Decay factor for feature fusion 
        """
        super(DAPFBlock, self).__init__()
        
        self.num_regions = num_regions
        self.in_channels = in_channels
        self.fusion_gamma = fusion_gamma
        
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.alpha = alpha
        self.rbf_sigma = rbf_sigma
        self.use_multi_kernel = use_multi_kernel
        
        # Learnable prior weights (pi_k) 
        self.prior_weights = nn.Parameter(torch.tensor([prior_weights[i] for i in range(num_regions)], dtype=torch.float32))

        # MMD Kernel Sigmas 
        self.sigma_list = [0.1, 0.5, 1.0, 2.0] if use_multi_kernel else [rbf_sigma]
            
        # Feature Reconstruction Convolution 
        self.region_conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        
        # Channel Modulator MLP 
        reduction = max(1, in_channels // 4)  
        self.channel_modulator = nn.Sequential(
            nn.Linear(in_channels, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, in_channels),
            nn.Softplus()  
        )
        
        # Orthogonal Decomposition Layers 
        # Decomposes input C into Structure (C_str) and Noise (C_noi) components.
        self.mid_channels = in_channels // 2
        self.struct_extractor = nn.Conv3d(in_channels, self.mid_channels, 1, bias=False)
        self.noise_extractor  = nn.Conv3d(in_channels, self.mid_channels, 1, bias=False)
        self.struct_up        = nn.Conv3d(self.mid_channels, in_channels, 1, bias=False)

    def compute_region_features(self, Z: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """
        Region-Specific Feature Aggregation.
        Aggregates features within each ROI mask.
        """
        B, C, D, H, W = Z.shape
        K = masks.shape[0]

        resized_masks = F.interpolate(masks.unsqueeze(1), size=(D, H, W), mode='nearest')  # [K, 1, D, H, W]
        
        region_features = []
        valid_masks = []
        
        for b in range(B):
            sample_features = []
            sample_valid = []
            
            for k in range(K):
                mask = resized_masks[k]  # [1, D, H, W]
                mask_valid = (mask.sum() > 0)
                sample_valid.append(mask_valid)
                
                if mask_valid:
                    masked_z = Z[b].unsqueeze(0) * mask  # [1, C, D, H, W]
                    # Average pooling within ROI 
                    region_feat = masked_z.sum(dim=[2,3,4]) / (mask.sum() + 1e-6)  # [1, C]
                    region_feat = region_feat.squeeze(0)
                else:
                    region_feat = torch.zeros(C).to(Z.device)
                
                sample_features.append(region_feat)
            
            region_features.append(torch.stack(sample_features))
            valid_masks.append(sample_valid)
        
        return torch.stack(region_features), valid_masks

    def compute_mmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD).
        Quantifies distribution shift between disease classes (AD vs CN).
        """
        if X.size(0) == 0 or Y.size(0) == 0:
            return torch.tensor(0.0).to(X.device) 
        m = X.size(0)
        n = Y.size(0)
        
        if self.use_multi_kernel:
            mmd2 = torch.zeros(len(self.sigma_list)).to(X.device)
            for idx, sigma in enumerate(self.sigma_list):
                # Calculate Kernel Matrices (Gaussian RBF) 
                XX = torch.cdist(X, X)**2 / (2 * sigma**2)
                YY = torch.cdist(Y, Y)**2 / (2 * sigma**2)
                XY = torch.cdist(X, Y)**2 / (2 * sigma**2)
                
                k_XX = torch.exp(-XX)
                k_YY = torch.exp(-YY)
                k_XY = torch.exp(-XY)
                
                # Unbiased MMD Statistic 
                term1 = (k_XX.sum() - torch.diag(k_XX).sum()) / (m * (m - 1)) if m > 1 else 0
                term2 = (k_YY.sum() - torch.diag(k_YY).sum()) / (n * (n - 1)) if n > 1 else 0
                term3 = k_XY.sum() * (-2 / (m * n))
                
                mmd2[idx] = term1 + term2 + term3
            return mmd2.mean()
        else:
            # Single Kernel 
            XX = torch.cdist(X, X)**2 / (2 * self.rbf_sigma**2)
            YY = torch.cdist(Y, Y)**2 / (2 * self.rbf_sigma**2)
            XY = torch.cdist(X, Y)**2 / (2 * self.rbf_sigma**2)
            
            k_XX = torch.exp(-XX)
            k_YY = torch.exp(-YY)
            k_XY = torch.exp(-XY)
            
            term1 = (k_XX.sum() - torch.diag(k_XX).sum()) / (m * (m - 1)) if m > 1 else 0
            term2 = (k_YY.sum() - torch.diag(k_YY).sum()) / (n * (n - 1)) if n > 1 else 0
            term3 = k_XY.sum() * (-2 / (m * n))
            
            return term1 + term2 + term3

    def compute_spatial_weights(self, region_features: torch.Tensor, 
                                valid_masks: List, masks: torch.Tensor,
                                labels: torch.Tensor, ages: torch.Tensor,
                                target_size: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prior-Guided Adaptive Weighting.
        Computes spatial attention map B based on MMD and anatomical priors.
        """
        B, K, feat_dim = region_features.shape
        device = region_features.device
        
        D_mmd = torch.zeros(K).to(device)
        max_mmd = torch.tensor(1e-6).to(device)
        
        normalized_mmd = torch.zeros(K).to(device)
        reg_loss = torch.tensor(0.0, device=device)
        
        # Calculate MMD if labels are available 
        if labels is not None:
            for k in range(K):
                valid_indices = [b for b in range(B) if valid_masks[b][k]]
                if len(valid_indices) < 2: continue
                    
                valid_labels = labels[valid_indices]
                ad_feats = region_features[valid_indices, k][valid_labels == 2]
                mci_feats = region_features[valid_indices, k][valid_labels == 1]
                cn_feats = region_features[valid_indices, k][valid_labels == 0]
                
                if ad_feats.size(0) == 0 or mci_feats.size(0) == 0 or cn_feats.size(0) == 0:
                    continue
                    
                mmd_ad_cn = self.compute_mmd(ad_feats, cn_feats)
                mmd_mci_ad = self.compute_mmd(mci_feats, ad_feats)
                D_mmd[k] = mmd_ad_cn + mmd_mci_ad
                
                if D_mmd[k] > max_mmd: max_mmd = D_mmd[k]
            
            normalized_mmd = (D_mmd / max_mmd.clamp(min=1e-6)).clamp(min=0.0)
            
            # Alignment Loss (KL Divergence) 
            # Regularizes learnable prior (pi_k) towards data-driven MMD distribution.
            p = F.softmax(self.prior_weights[:K], dim=0)
            q = F.softmax(normalized_mmd, dim=0)
            reg_loss = 0.5 * F.kl_div(p.log(), q, reduction='batchmean') + 0.5 * F.kl_div(q.log(), p, reduction='batchmean')
        
        # Fuse Prior and MMD 
        base_weights = (self.lambda_param * self.prior_weights[:K] + (1 - self.lambda_param) * normalized_mmd) 
        
        # Age Modulation 
        age_factor = 1 + F.softplus(self.alpha * (ages - 50))
        W_k = torch.clamp(base_weights.unsqueeze(0).expand(B, -1) * age_factor.view(B, 1), min=0.0, max=self.gamma)  
        W_k = F.softmax(W_k, dim=1) 
        
        # Project weights back to spatial domain 
        resized_masks = F.interpolate(masks.unsqueeze(1), size=target_size, mode='nearest').squeeze(1)  # [K, D, H, W]
        spatial_weights = torch.einsum('bk,kdhw -> bdhw', W_k, resized_masks).unsqueeze(1)
        
        return spatial_weights, reg_loss

    def feature_reconstruction(self, Z: torch.Tensor, region_features: torch.Tensor, 
                              masks: torch.Tensor, spatial_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Region-Channel Modulation 
        """
        B, C, D, H, W = Z.shape
        K = masks.shape[0] 
        resized_masks = F.interpolate(masks.unsqueeze(1), size=(D, H, W), mode='nearest').squeeze(1)
        
        # Orthogonal Decomposition 
        cond_struct = self.struct_extractor(Z)  # Structure [B, C/2, D, H, W]
        cond_noise = self.noise_extractor(Z)    # Noise [B, C/2, D, H, W]
        
        s = cond_struct.flatten(2)
        n = cond_noise.flatten(2)
        
        cos_sim = F.cosine_similarity(s, n, dim=2, eps=1e-8)
        orthogonal_term = cos_sim.abs().mean()
    
        
        ortho_loss = config.lambda_ortho * orthogonal_term
        
        
        # Use structural component for reconstruction 
        Z = self.struct_up(cond_struct)
        
        # Topology-Aware Reconstruction 
        p_k = region_features.permute(0, 2, 1)  # [B, C, K]
        # Interpolate region features to spatial map 
        p_k_up = F.interpolate(p_k.reshape(B * K, C, 1, 1, 1), size=(D, H, W), mode='trilinear', align_corners=True)
        p_k_up = p_k_up.reshape(B, K, C, D, H, W).permute(0, 2, 3, 4, 5, 1)  # [B, C, D, H, W, K]
        
        Z_masked = Z.unsqueeze(2) * resized_masks.unsqueeze(0).unsqueeze(1)  # Masked Z
        conv_out = self.region_conv(Z_masked.reshape(B * K, C, D, H, W)).reshape(B, C, K, D, H, W).permute(0, 1, 3, 4, 5, 2)
        
        Z_hat = p_k_up + conv_out # Initial reconstructed features
        
        # Channel Modulation 
        modulation_vec = self.channel_modulator(region_features.reshape(B * K, C)).reshape(B, K, C)
        modulation_vec = modulation_vec.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(1, 5) # Broadcast
        Z_tilde = Z_hat * modulation_vec
        Z_sum = Z_tilde.sum(dim=-1)  # Sum over regions [B, C, D, H, W]
        
        Z_weighted = Z * torch.exp(-self.fusion_gamma * spatial_weights)
        Z_out = Z_sum + Z_weighted
        
        return Z_out, ortho_loss

    def forward(self, 
                Z: torch.Tensor, 
                masks: torch.Tensor,
                labels: torch.Tensor,
                ages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        D, H, W = Z.shape[2:]
        
        # 1. Feature Aggregation 
        region_features, valid_masks = self.compute_region_features(Z, masks)
        
        # 2. Adaptive Weighting 
        spatial_weights, reg_loss = self.compute_spatial_weights(region_features, valid_masks, masks, labels, ages, (D, H, W))
        
        # 3. Modulation and Fusion 
        Z_out, ortho_loss = self.feature_reconstruction(Z, region_features, masks, spatial_weights)
        
        total_reg_loss = reg_loss + ortho_loss
        

        return region_features, spatial_weights, Z_out, total_reg_loss
