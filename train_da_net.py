import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils.utils import seed_torch, save_checkpoint, load_checkpoint, write_csv_header
from utils.config import config
from model.da_net import DA_NET3D  
from dataset.adni_dataset import ThreeClassDataset

import warnings
warnings.filterwarnings("ignore")

"""
Usage:
    torchrun --nproc_per_node=<num_gpus> train_da_net.py
"""

def setup_distributed():
    """Initialize distributed training (NCCL)."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def train_da_net(rank: int, local_rank: int):
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        result_dir = os.path.join(config.exp_root, config.exp_da)
        os.makedirs(result_dir, exist_ok=True)
        loss_csv = os.path.join(result_dir, "loss_curve.csv")
        val_csv = os.path.join(result_dir, "validation.csv")
        write_csv_header(loss_csv, ["Epoch", "CrossEntropyLoss"])  
        write_csv_header(val_csv, ["Epoch", "Accuracy"]) 

    model = DA_NET3D(num_classes=config.num_classes).to(device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)

    if os.path.exists(config.CHECKPOINT_DA):
        load_checkpoint(config.CHECKPOINT_DA, model, optimizer, config.learning_rate, device)
        
    class_counts = torch.tensor([277, 427, 144], dtype=torch.float32)
    class_weights = 1.0 / (class_counts / class_counts.sum())
    class_weights = class_weights.to(device)
    
    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    
    best_acc = 0.0
    early_stop_patience = 50  
    no_improve = 0

    for epoch in range(1, config.epochs + 1):
        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch}/{config.epochs}")
        model.train()

        # Dataset (MRI Only)
        train_ds = ThreeClassDataset(root_MRI=config.train_FDG_MRI, stage="train", csv_path=config.train_FDG_CSV)
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, 
                            num_workers=config.numworker, pin_memory=True, drop_last=True)
        train_sampler.set_epoch(epoch)
        
        running_loss = 0.0

        for mri, label, _, _ in tqdm(loader, desc="Training", disable=(rank!=0)):
            mri = mri.unsqueeze(1).to(device)
            label = label.to(device)
            scores, _ = model(mri) 
            
            loss = criterion_class(scores, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        scheduler.step()
        
        if rank == 0:
            avg_loss = running_loss / len(loader)
            with open(loss_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, round(avg_loss, 6)])

        model.eval()
        all_preds = []
        all_labels = []
        
        val_ds = ThreeClassDataset(root_MRI=config.val_FDG_MRI, stage="val", csv_path=config.val_FDG_CSV)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.numworker, pin_memory=True)
        
        with torch.no_grad():
            for mri, label, _, _ in tqdm(val_loader, desc="Validation", disable=(rank!=0)):
                mri = mri.unsqueeze(1).to(device)
                label = la