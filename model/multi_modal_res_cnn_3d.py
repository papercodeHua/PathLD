import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import config

class ResCNN3DBackbone(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        # 3D CNN Backbone with residual connections
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.shortcut2 = nn.Conv3d(16, 32, kernel_size=1, stride=2, padding=0)  # Projection shortcut for res2
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.shortcut3 = nn.Conv3d(32, 64, kernel_size=1, stride=2, padding=0)  # Projection shortcut for res3
        
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.shortcut4 = nn.Conv3d(64, 64, kernel_size=1, stride=2, padding=0)  # Projection shortcut for res4 (even though channels same, stride=2)
        
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(64)
        self.shortcut5 = nn.Conv3d(64, 64, kernel_size=1, stride=2, padding=0)  # Projection shortcut for res5

        self.dropout = nn.Dropout3d(p=dropout)

        self.global_pool = nn.AdaptiveAvgPool3d(1)  # [B, 64, 1, 1, 1]

        self.apply(self._init_weights)  
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x, return_features=False):
        # Layer 1
        x1 = F.relu(self.bn1(self.conv1(x)))
        
        # Layer 2
        id2 = self.shortcut2(x1)
        x2 = F.relu(self.bn2(self.conv2(x1)) + id2)
        
        # Layer 3
        id3 = self.shortcut3(x2)
        x3 = F.relu(self.bn3(self.conv3(x2)) + id3)
        
        # Layer 4
        id4 = self.shortcut4(x3)
        x4 = F.relu(self.bn4(self.conv4(x3)) + id4)
        
        # Layer 5 (Final Conv)
        id5 = self.shortcut5(x4)
        x5 = F.relu(self.bn5(self.conv5(x4)) + id5) # [B, 64, D/16, H/16, W/16]
        
        features = self.dropout(x5)
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1).squeeze(-1)

        if return_features:
            return [x2, x3, x5, pooled]
        else:
            return pooled

class MultiModalResCNN3D(nn.Module):
    def __init__(self, num_classes=3, dropout=0.1, fusion_hidden=128):
        super().__init__()
        self.num_classes = num_classes

        self.mri_backbone = ResCNN3DBackbone(dropout=dropout)
        self.pet_backbone = ResCNN3DBackbone(dropout=dropout)

        self.fusion_head = nn.Sequential(
            nn.Linear(64 * 2, fusion_hidden),  # 128 -> hidden
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)  # hidden -> num_classes
        )

        self.apply(self._init_weights)  
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def extract_features(self, mri, pet):

        pet_feats = self.pet_backbone(pet, return_features=True)
        
        mri_pooled = self.mri_backbone(mri, return_features=False)
        
        fused_vec = torch.cat([mri_pooled, pet_feats[-1]], dim=1) # [B, 128]
        
        return pet_feats[:-1] + [fused_vec]
            
    def forward(self, mri, pet):
        mri_pooled = self.mri_backbone(mri)
        
        pet_pooled = self.pet_backbone(pet)
        
        fused = torch.cat([mri_pooled, pet_pooled], dim=1)  # [B, 128]
        
        class_scores = self.fusion_head(fused)  # [B, num_classes]

        probabilities = F.softmax(class_scores, dim=1)

        return probabilities