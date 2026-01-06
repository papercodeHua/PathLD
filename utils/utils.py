import torch
import torch.nn as nn
import numpy as np
import random
import os
import csv
from typing import Optional

def weights_init(m: nn.Module):
    """
    Initialize network weights (Normal distribution for Conv/BN).
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, filename: str = "my_checkpoint.pth.tar"):
    """
    Save model checkpoint. Supports DDP/DataParallel models.
    """
    print("=> Saving checkpoint")
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    checkpoint = {
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer], lr: float, device: torch.device):
    """
    Load model checkpoint. Handles 'module.' prefix for DDP compatibility.
    """
    print(f"=> Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    state_dict = checkpoint["state_dict"]
    
    # Handle DDP prefix mismatch 
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        if not list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k 
                new_state_dict[name] = v
            state_dict = new_state_dict
    else:
        if list(state_dict.keys())[0].startswith('module.'):
            # Remove 'module.' prefix / 移除前缀
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] 
                new_state_dict[name] = v
            state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def seed_torch(seed: int = 0):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def write_csv_header(file_path: str, header: list):
    """
    Write CSV header if file does not exist.
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)      

    
   
