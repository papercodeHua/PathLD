import torch
import torch.nn as nn
import torch.nn.functional as F

class DA_NET3D(nn.Module):
    """
    Diagnosis-Aware Network (DA-Net).
    A lightweight 3D classifier for generating Saliency Maps and Pseudo-labels.
    """
    def __init__(self, num_classes: int = 3, embed_dim: int = 128, dropout: float = 0.1):  
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim 

        # 3D CNN Backbone
        self.conv1 = nn.Conv3d(1, embed_dim // 4, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(embed_dim // 4)
        self.conv2 = nn.Conv3d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(embed_dim // 2)
        self.conv3 = nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(embed_dim)
        
        # Final Conv Layer for CAM 
        self.conv4 = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(embed_dim)

        self.dropout = nn.Dropout3d(p=dropout)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.class_head = nn.Linear(embed_dim, num_classes)

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature Extraction 
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.relu(self.bn3(self.conv3(x))) 
        features = self.bn4(self.conv4(x))  # Last spatial features [B, C, D, H, W]
        features = F.relu(features)
        features_drop = self.dropout(features)

        pooled = self.global_pool(features_drop).squeeze(-1).squeeze(-1).squeeze(-1) # [B, C]
        class_scores = self.class_head(pooled) # [B, num_classes]

        # Generate Saliency Map (CAM) 
        # Weights: [num_classes, C] -> [num_classes, C, 1, 1, 1]
        weights = self.class_head.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Weighted sum of features: [B, C, D, H, W] * [K, C, 1, 1, 1] -> [B, K, D, H, W]
        cam_maps = torch.einsum('bcdhw,kc...->bkdhw', features, weights)
        
        cam_maps = (cam_maps - cam_maps.min()) / (cam_maps.max() - cam_maps.min() + 1e-8)
        saliency_maps = torch.sigmoid(cam_maps)

        saliency_maps = F.interpolate(saliency_maps, size=(160, 192, 160), mode='trilinear', align_corners=False)

        return class_scores, saliency_maps