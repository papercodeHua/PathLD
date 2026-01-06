import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import *
from model.multi_modal_res_cnn_3d import MultiModalResCNN3D
from dataset.ADNI_dataset import RealMultiModalDataset
from utils.config import config
import csv
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

warnings.filterwarnings("ignore")

# torchrun --nproc_per_node=<num_gpus> train_mul_res_cnn.py

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def specificity(y_true, y_pred, classes):
    specs = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specs.append(spec)
    return np.mean(specs)

def train_ResCNN(rank, local_rank):
    device = torch.device(f"cuda:{local_rank}")

    result_dir = os.path.join("result", config.exp_mri_pet)  
    if rank == 0:
        os.makedirs(result_dir, exist_ok=True)
        loss_csv = os.path.join(result_dir, "loss_curve.csv")
        val_csv = os.path.join(result_dir, "validation.csv")
        write_csv_header(loss_csv, ["Epoch", "class_loss"])
        write_csv_header(val_csv, ["Epoch", "ACC", "AUC", "SEN", "SPE", "F1S"])

    model = MultiModalResCNN3D(num_classes=3).to(device)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-5
    )


    if config.CHECKPOINT_ResCNN and os.path.exists(config.CHECKPOINT_ResCNN):
        load_checkpoint(config.CHECKPOINT_ResCNN, model, optimizer, config.learning_rate, device)

    class_counts = torch.tensor([277, 427, 144], dtype=torch.float32)
    class_weights = 1.0 / (class_counts / class_counts.sum())
    class_weights = class_weights.to(device)

    criterion_class = nn.NLLLoss(weight=class_weights)

    best_acc = 0.0
    early_stop_patience = 50
    no_improve = 0

    for epoch in range(1, config.epochs + 1):
        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch}/{config.epochs}")
        model.train()

        train_ds = RealMultiModalDataset(
            root_MRI=config.train_FDG_MRI,   
            root_FDG=config.train_FDG,   
            stage="train",
            csv_path=config.train_FDG_CSV  #
        )
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.numworker,
            pin_memory=True,
            drop_last=True
        )
        train_sampler.set_epoch(epoch)

        running_class_loss = 0.0

        for mri, fdg, label, age, _ in tqdm(loader, desc="Training", disable=(rank != 0)):
            # mri, fdg: [B, D, H, W] → [B, 1, D, H, W]
            mri = np.expand_dims(mri, axis=1)
            mri = torch.tensor(mri).to(device)
            fdg = np.expand_dims(fdg, axis=1)
            fdg = torch.tensor(fdg).to(device)

            # label: [B]，0=CN, 1=MCI, 2=AD
            label = label.to(device)

            probabilities = model(mri, fdg)  # [B, 3]
            log_probs = torch.log(probabilities + 1e-8)
            class_loss = criterion_class(log_probs, label)
            loss = class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_class_loss += class_loss.item()

        scheduler.step()
        if rank == 0:
            avg_class_loss = running_class_loss / len(loader)
            with open(loss_csv, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, round(avg_class_loss, 6)])

        if rank == 0:
            model.module.eval()
            all_preds = []
            all_labels = []
            all_probs = []

            val_ds = RealMultiModalDataset(
                root_MRI=config.test_FDG_MRI,   
                root_FDG=config.test_FDG,   
                stage="val",
                csv_path=config.test_FDG_CSV  
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=config.numworker,
                pin_memory=True
            )

            with torch.no_grad():
                for mri, fdg, label, age, _ in tqdm(val_loader, desc="Validation"):
                    mri = np.expand_dims(mri, axis=1)   # [1, D, H, W] → [1, 1, D, H, W]
                    mri = torch.tensor(mri).to(device)
                    fdg = np.expand_dims(fdg, axis=1)
                    fdg = torch.tensor(fdg).to(device)
                    label = label.to(device)          

                    probabilities = model.module(mri, fdg)  # [1, 3]
                    preds = probabilities.argmax(dim=1)  # [1]

                    all_preds.append(preds.cpu().item())
                    all_probs.append(probabilities.cpu().numpy())
                    all_labels.append(label.cpu().item())

            all_probs = np.vstack(all_probs)   # [N, 3]
            all_preds = np.array(all_preds)    # [N]
            all_labels = np.array(all_labels)  # [N]

            acc = accuracy_score(all_labels, all_preds)

            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

            _, sen, f1s, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro'
            )

            spe = specificity(all_labels, all_preds, classes=[0, 1, 2])
            with open(val_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch,
                    round(acc, 4),
                    round(auc, 4),
                    round(sen, 4),
                    round(spe, 4),
                    round(f1s, 4)
                ])

            if acc > best_acc:
                best_acc = acc
                save_checkpoint(model, optimizer, filename=config.CHECKPOINT_ResCNN)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        dist.barrier()

if __name__ == '__main__':
    seed_torch()
    local_rank, rank = setup_distributed()
    train_ResCNN(rank, local_rank)
    cleanup()