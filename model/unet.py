import math
import torch
from torch import nn
from utils.config import config
from inspect import isfunction
from model.blocks import GroupNorm, Upsample, Downsample, ResidualBlock, Swish, PFBlock 
import nibabel as nib
import numpy as np
import os
import glob
from dataset.ADNI_dataset import crop
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    if exists(val): return val
    return d() if isfunction(d) else d

class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class ResnetBlocWithAttn(nn.Module):
    """
    ResNet Block with optional Self-Attention.
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    """
    Denoising UNet with DAPF Block conditioning.
    """
    def __init__(
        self,
        in_channel=2, # Noisy Latent + Condition Feature 
        out_channel=1,
        inner_channel=128,
        norm_groups=32,
        channel_mults=(1, 2, 2, 4),
        attn_res=(4,),
        res_blocks=2,
        dropout=0,
        with_time_emb=True,
        image_size=40
    ):
        super().__init__()
        
        # Load Atlas Masks 
        masks_tensor = self.load_atlas_masks(config.atlas_dir)
        self.register_buffer('masks', masks_tensor)
        
        # Time Embedding MLP 
        if with_time_emb:
            time_dim = config.time_dim
            self.time_mlp = nn.Sequential(
                TimeEmbedding(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                Swish(),
                nn.Linear(time_dim * 4, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        
        # Initial Conv
        downs = [nn.Conv3d(in_channel, inner_channel, kernel_size=3, padding=1)]
        
        # Downsampling Path
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        # Middle Path
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups, 
                                dropout=dropout, with_attn=False)
        ])

        # Upsampling Path
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=time_dim, dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2
        self.ups = nn.ModuleList(ups)
        
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        
        # Condition Encoder (Cond-Net) 
        self.condition = nn.Sequential(
            nn.Conv3d(1, 16, 4, 2, 1),
            ResidualBlock(16, 16),
            GroupNorm(16),
            Swish(),
            nn.Conv3d(16, 16, 4, 2, 1),
        )
        
        # Class Label Embedding 
        self.label_emb = nn.Embedding(config.num_classes, time_dim)
        
        # PF Block ( Pathology-Focused)
        self.PFBlock = PFBlock(
            num_regions=config.num_regions,
            in_channels=config.in_channels,
            prior_weights=config.prior_weights,
            lambda_param=config.lambda_param,
            gamma= config.gamma,
            rbf_sigma= config.rbf_sigma,
            alpha= config.alpha,
            use_multi_kernel= config.use_multi_kernel,
            fusion_gamma= config.fusion_gamma
        )
        
        self.g_roi = 0.3  # Residual gating factor 
        
        # Condition Adapter (1x1 conv to match channels if needed) 
        self.cond_adapter = nn.Sequential(
            nn.InstanceNorm3d(16, eps=1e-6, affine=True),
            Swish(),
            nn.Conv3d(16, 1, kernel_size=1, bias=False)
        )

    def load_atlas_masks(self, dir_path):
        """Load anatomical masks for ROI-based attention."""
        mask_files = sorted(glob.glob(os.path.join(dir_path, "*.nii.gz")))
        if len(mask_files) != config.num_regions:
            raise ValueError(f"Expected {config.num_regions} mask files, but found {len(mask_files)} in {dir_path}")
        masks_list = []
        for file in mask_files:
            data = nib.load(file).get_fdata().astype(np.float32)
            mid = crop(data) 
            masks_list.append(torch.from_numpy(mid))
        return torch.stack(masks_list)

    def build_B(self, D, H, W, label, saliency):
        """
        Build Lesion Guidance Map B.
        Combines Atlas Priors + Saliency Map.
        """
        K = self.masks.shape[0]
        masks_resized = F.interpolate(self.masks.unsqueeze(1), size=(D, H, W), mode='nearest').squeeze(1)

        # Use learnable prior weights (softmax normalized) 
        pw_live = torch.softmax(self.PFBlock.prior_weights[:K], dim=0)     

        B_roi = torch.einsum('kdhw,k->dhw', masks_resized, pw_live).unsqueeze(0).unsqueeze(0)
        B_roi = B_roi / (B_roi.amax(dim=(2,3,4), keepdim=True) + 1e-6)                    

        if saliency is None:
            return B_roi

        S = saliency
        if S.dim() == 4: S = S.unsqueeze(1)
        S = F.interpolate(S, size=(D, H, W), mode='trilinear', align_corners=False)
        
        if label is not None:
            cls_map = S[torch.arange(S.shape[0]), label.long()].unsqueeze(1)
        else:
            cls_map = S.mean(dim=1, keepdim=True)
        alpha, gamma_s = 2.0, 1.0 
        cls_map = cls_map / (cls_map.abs().amax(dim=(2,3,4), keepdim=True) + 1e-6)
        S_refine = torch.sigmoid(alpha * cls_map).pow(gamma_s)

        return (B_roi * S_refine)

    def forward(self, x, y, time, label=None, ages=None, saliency=None):
        """
        Forward pass of DAPF-LDM UNet.
        x: Noisy Latent [B, 1, D, H, W]
        y: MRI Volume [B, 1, D, H, W]
        time: Timesteps [B]
        label: Diagnostic Labels [B]
        """
        B = x.shape[0]
        
        # 1. Condition Extraction (Cond-Net) 
        cond = self.condition(y)

        # 2. PF Modulation 
        # Returns: region_feats, weights, z_out (modulated feature), reg_loss
        _, _, z, reg_loss = self.PFBlock(cond, self.masks, label, ages)
        
        # 3. Lesion Guidance Map B Construction 
        D, H, W = z.shape[2:]
        B_map = self.build_B(D, H, W, label, saliency)
        if B_map.shape[0] == 1:
            B_map = B_map.expand(B, -1, -1, -1, -1)

        # 4. Residual Fusion 
        cond_guided = cond + self.g_roi * (z * B_map)
        cond_fused = self.cond_adapter(cond_guided)

        # 5. Time & Label Embedding 
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        if label is not None:
            k = self.label_emb(label.squeeze().long())
            t = t + k if t is not None else k
            
        # 6. UNet Backbone 
        x = torch.cat([x, cond_fused], dim=1) # Concatenate noisy latent + condition

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        out = self.final_conv(x)  
        
        if self.training:
            return out, reg_loss, cond_fused
        return out

class EMA:
    """ Exponential Moving Average for model parameters. """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None: return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):

        ema_model.load_state_dict(model.state_dict())

