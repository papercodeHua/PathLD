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

# -------------------------------------------------------------------------
# Pathology-Focused (PF) Block
# -------------------------------------------------------------------------

class PFBlock(nn.Module):
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
        super(PFBlock, self).__init__()
        
        self.num_regions = num_regions
        self.in_channels = in_channels
        self.fusion_gamma = fusion_gamma
        
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.alpha = alpha
        self.rbf_sigma = rbf_sigma
        self.use_multi_kernel = use_multi_kernel
        
        # Learnable prior weights (pi_k) 
        # Using .clone().detach() ensures we don't track history for the initialization tensor
        weights_tensor = torch.tensor([prior_weights.get(i, 1.0) for i in range(num_regions)], dtype=torch.float32)
        self.prior_weights = nn.Parameter(weights_tensor)

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

        # Ensure masks are resized to feature map dimensions
        # Assuming masks input is [K, D_orig, H_orig, W_orig], we resize to [D, H, W]
        resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=(D, H, W), mode='nearest')  # [K, 1, D, H, W]
        
        region_features = []
        valid_masks = []
        
        for b in range(B):
            sample_features = []
            sample_valid = []
            
            for k in range(K):
                mask = resized_masks[k]  # [1, D, H, W]
                # Check if region exists in this sample (non-empty mask)
                mask_sum = mask.sum()
                mask_valid = (mask_sum > 0)
                sample_valid.append(mask_valid.item())
                
                if mask_valid:
                    masked_z = Z[b].unsqueeze(0) * mask  # [1, C, D, H, W]
                    # Average pooling within ROI 
                    region_feat = masked_z.sum(dim=[2,3,4]) / (mask_sum + 1e-6)  # [1, C]
                    region_feat = region_feat.squeeze(0)
                else:
                    region_feat = torch.zeros(C).to(Z.device)
                
                sample_features.append(region_feat)
            
            region_features.append(torch.stack(sample_features)) # [K, C]
            valid_masks.append(sample_valid)
        
        return torch.stack(region_features), valid_masks # [B, K, C]

    def compute_mmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD).
        Quantifies distribution shift between disease classes.
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
        Computes spatial attention map based on Multi-Class MMD (Eq. 6) and anatomical priors.
        """
        B, K, feat_dim = region_features.shape
        device = region_features.device
        
        D_mmd = torch.zeros(K).to(device)
        max_mmd = torch.tensor(1e-6).to(device)
        
        reg_loss = torch.tensor(0.0, device=device)
        
        # Calculate MMD if labels are available 
        if labels is not None:
            for k in range(K):
                # Filter valid samples in the batch for region k
                valid_indices = [b for b in range(B) if valid_masks[b][k]]
                if len(valid_indices) < 2: continue
                
                # Extract features for each class within the valid samples
                # Assuming labels: 0=CN, 1=MCI, 2=AD
                current_labels = labels[valid_indices]
                current_feats = region_features[valid_indices, k]
                
                cn_feats = current_feats[current_labels == 0]
                mci_feats = current_feats[current_labels == 1]
                ad_feats = current_feats[current_labels == 2]
                
                if ad_feats.size(0) == 0 or mci_feats.size(0) == 0 or cn_feats.size(0) == 0:
                    continue
                
                # 1. AD vs CN
                mmd_ad_cn = self.compute_mmd(ad_feats, cn_feats)
                # 2. MCI vs CN (Missing in original code)
                mmd_mci_cn = self.compute_mmd(mci_feats, cn_feats) 
                # 3. AD vs MCI
                mmd_ad_mci = self.compute_mmd(ad_feats, mci_feats)
                
                # Total Distributional Shift: Sum of all pairwise distances
                D_mmd[k] = mmd_ad_cn + mmd_mci_cn + mmd_ad_mci
                
                if D_mmd[k] > max_mmd: max_mmd = D_mmd[k]
            
            # Normalize Data-driven Evidence
            normalized_mmd = (D_mmd / max_mmd.clamp(min=1e-6)).clamp(min=0.0)
            
            # Alignment Loss (KL Divergence) 
            # Regularizes learnable prior (pi_k) towards data-driven MMD distribution.
            p = F.softmax(self.prior_weights[:K], dim=0)
            q = F.softmax(normalized_mmd, dim=0)
            # Symmetric KL Divergence for stability
            reg_loss = 0.5 * F.kl_div(p.log(), q, reduction='batchmean') + 0.5 * F.kl_div(q.log(), p, reduction='batchmean')
        else:
            # If no labels (inference), rely solely on Prior
            normalized_mmd = torch.zeros(K).to(device)
        
        base_weights = (self.lambda_param * self.prior_weights[:K] + (1 - self.lambda_param) * normalized_mmd) 
        
        # Age Modulation 
        # ages: [B], expand to [B, K]
        age_factor = 1 + F.softplus(self.alpha * (ages - 50))
        
        # Compute Unnormalized Weights
        W_k_unnorm = base_weights.unsqueeze(0).expand(B, -1) * age_factor.view(B, 1)
        
        # Clamp and Softmax
        W_k_clamped = torch.clamp(W_k_unnorm, min=0.0, max=self.gamma)  
        W_k = F.softmax(W_k_clamped, dim=1) 
        
        # Project weights back to spatial domain 
        resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1)  # [K, D, H, W]
        # Weighted sum of masks: [B, K] * [K, D, H, W] -> [B, D, H, W]
        spatial_weights = torch.einsum('bk,kdhw -> bdhw', W_k, resized_masks).unsqueeze(1) # [B, 1, D, H, W]
        
        return spatial_weights, reg_loss

    def feature_reconstruction(self, Z: torch.Tensor, region_features: torch.Tensor, 
                              masks: torch.Tensor, spatial_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Region-Channel Modulation with Orthogonal Decomposition
        """
        B, C, D, H, W = Z.shape
        K = masks.shape[0] 
        resized_masks = F.interpolate(masks.unsqueeze(1).float(), size=(D, H, W), mode='nearest').squeeze(1)
        
        # Orthogonal Decomposition 
        cond_struct = self.struct_extractor(Z)  # Structure [B, C/2, D, H, W]
        cond_noise = self.noise_extractor(Z)    # Noise [B, C/2, D, H, W]
        
        # Orthogonality Loss
        s = cond_struct.flatten(2)
        n = cond_noise.flatten(2)
        
        # Minimize absolute cosine similarity
        cos_sim = F.cosine_similarity(s, n, dim=2, eps=1e-8)
        orthogonal_term = cos_sim.abs().mean()
        
        # Use structural component for reconstruction 
        # We project back to full channels
        Z_clean = self.struct_up(cond_struct)
        
        # Region-based Feature Projection (Topology-Aware)
        # Project vector v_k back to spatial map
        p_k = region_features.permute(0, 2, 1)  # [B, C, K]
        
        # Let's align with the previous code's logic which seemed to mix pixel-level conv with region-level features.
        
        # 1. Spatially broadcasting region features to their masks
        # [B, C, K] -> [B, C, D, H, W] via masks
        Z_region_base = torch.zeros_like(Z_clean)
        for k in range(K):
             # region_features: [B, K, C]
             feat_k = region_features[:, k, :].view(B, C, 1, 1, 1)
             mask_k = resized_masks[k].unsqueeze(0).unsqueeze(1) # [1, 1, D, H, W]
             Z_region_base += feat_k * mask_k
             
        # 2. Refine these region features with a convolution (phi function in paper)
        Z_rec_features = self.region_conv(Z_region_base) # [B, C, D, H, W]
        
        # 3. Channel Modulation
        # We compute modulation vector per region: [B, K, C]
        mod_vecs = self.channel_modulator(region_features.reshape(B * K, C)).reshape(B, K, C)
        
        # Apply modulation spatially
        Z_modulated = torch.zeros_like(Z_clean)
        for k in range(K):
            mask_k = resized_masks[k].unsqueeze(0).unsqueeze(1)
            mod_k = mod_vecs[:, k, :].view(B, C, 1, 1, 1)
            # Modulate the reconstructed features
            Z_modulated += Z_rec_features * mod_k * mask_k
            
        # 4. Final Fusion with Residual
        # Let's stick closer to the paper's residual formulation:
        Z_out = Z_clean + self.fusion_gamma * (Z_modulated * spatial_weights)
        
        return Z_out, orthogonal_term

    def forward(self, 
                Z: torch.Tensor, 
                masks: torch.Tensor,
                labels: torch.Tensor,
                ages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        D, H, W = Z.shape[2:]
        
        # 1. Feature Aggregation 
        region_features, valid_masks = self.compute_region_features(Z, masks)
        
        # 2. Adaptive Weighting 
        # Computes W_k and converts to spatial map B (spatial_weights)
        spatial_weights, reg_loss = self.compute_spatial_weights(region_features, valid_masks, masks, labels, ages, (D, H, W))
        
        # 3. Modulation and Fusion 
        Z_out, ortho_loss = self.feature_reconstruction(Z, region_features, masks, spatial_weights)
        
        total_reg_loss = reg_loss + ortho_loss
        
        return region_features, spatial_weights, Z_out, total_reg_loss


