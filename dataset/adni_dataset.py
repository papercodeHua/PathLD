import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Union
from utils.config import config

def nifti_to_numpy(file_path: str) -> np.ndarray:
    """
    Load a NIfTI file and convert it to a float32 numpy array.
    """
    data = nib.load(file_path).get_fdata()
    return data.astype(np.float32)

def random_translation(data: np.ndarray) -> np.ndarray:
    """
    Apply random translation augmentation.
    Assumes input size is larger than target size (160, 192, 160).
    Randomly selects a window within the margin.
    
    Args:
        data: Input 3D volume (numpy array).
    Returns:
        Cropped 3D volume of size (160, 192, 160).
    """
    # Shift range: [-2, 2]
    i, j, z = np.random.randint(-2, 3, size=3)
    # Target size: 160x192x160
    # Fixed offsets (10, 18, 10) likely account for background margins in raw ADNI data.
    return data[10+i:170+i, 18+j:210+j, 10+z:170+z]

def crop(data: np.ndarray) -> np.ndarray:
    """
    Center crop (or fixed position crop) to target size.
    Used for validation/testing to ensure deterministic input.
    Target Size: (160, 192, 160)
    """
    return data[10:170, 18:210, 10:170]

class OneDataset(Dataset):
    """
    Loads PET images only.
    """
    def __init__(self, root_FDG: str, stage: str = "train"):
        self.root_FDG = root_FDG
        self.image_files = sorted(glob.glob(os.path.join(self.root_FDG, "*.nii.gz")))
        self.image_ids = [os.path.basename(f) for f in self.image_files]
        self.stage = stage

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, str]:
        file_path = self.image_files[index]
        sample_id = self.image_ids[index]
        
        FDG = nifti_to_numpy(file_path)
        if self.stage == "train":
            FDG = random_translation(FDG)
        else:
            FDG = crop(FDG)
            
        # Min-Max Normalization 
        # Using 1e-8 to prevent division by zero 
        FDG = (FDG - FDG.min()) / (FDG.max() - FDG.min() + 1e-8)
        
        return FDG, sample_id

class TwoDataset(Dataset):
    """
    Dataset for Stage II: Latent Diffusion Training.
    Loads paired MRI and PET images based on Subject ID and Scan ID.
    """
    def __init__(self, root_MRI: str, root_FDG: str, stage: str = "train", 
                 csv_path: str = "./datas/FDG/MRI/FDG_train_MRI_info.csv"):
        self.mri_dir = root_MRI
        self.fdg_dir = root_FDG
        self.stage = stage
        
        # Load labels and demographic info 
        self.df_labels = pd.read_csv(csv_path, encoding="ISO-8859-1")
        self.group_mapping = {"CN": 0, "MCI": 1, "AD": 2}

        # Match files 
        mri_paths = sorted(glob.glob(os.path.join(self.mri_dir, "*.nii.gz")))
        fdg_paths = sorted(glob.glob(os.path.join(self.fdg_dir, "*.nii.gz")))

        def extract_pair_id(path):
            basename = os.path.basename(path).replace(".nii.gz", "")
            parts = basename.split("-")
            subject = parts[0]
            scan_id = int(parts[-1])  
            return subject, scan_id

        mri_dict = {extract_pair_id(p): p for p in mri_paths}
        fdg_dict = {extract_pair_id(p): p for p in fdg_paths}

        # Find intersection of keys 
        common_keys = sorted(set(mri_dict.keys()) & set(fdg_dict.keys()),
                            key=lambda k: (k[0], k[1]))

        self.pairs = []
        for subject, scan_id in common_keys:
            self.pairs.append((mri_dict[(subject, scan_id)],
                            fdg_dict[(subject, scan_id)],
                            subject, scan_id))  
            
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, str, torch.Tensor, torch.Tensor]:
        mri_path, fdg_path, subject, scan_id = self.pairs[index]  

        # Load and Preprocess 
        mri = nifti_to_numpy(mri_path)
        mri = crop(mri) 
        
        fdg = nifti_to_numpy(fdg_path)
        if self.stage != "train":
            fdg = crop(fdg)
        
        # Get Label and Age 
        row = self.df_labels[(self.df_labels["Subject"] == subject) & (self.df_labels["Image_ID"] == scan_id)]
        if row.empty:
            raise ValueError(f"No matching row found for Subject={subject}, Image_ID={scan_id} in CSV.")
        
        label_str = row["Group"].values[0]
        label = torch.tensor(self.group_mapping[label_str], dtype=torch.float32)
        age = torch.tensor(row["Age"].values[0], dtype=torch.float32)

        unique_sample_id = f"{subject}_{scan_id}"

        return mri, fdg, unique_sample_id, label, age

class RealMultiModalDataset(Dataset):
    """
    Dataset for Multi-Modal Classifier (ResCNN) Training.
    Used for Fusion-Perceptual Loss training.
    """
    def __init__(self, root_MRI: str, root_FDG: str, stage: str = "train", 
                 csv_path: str = "./datas/FDG/MRI/FDG_train_MRI_info.csv"):
        self.mri_dir = root_MRI
        self.fdg_dir = root_FDG
        self.stage = stage
        self.df_labels = pd.read_csv(csv_path, encoding="ISO-8859-1")
        self.group_mapping = {"CN": 0, "MCI": 1, "AD": 2}

        mri_paths = sorted(glob.glob(os.path.join(self.mri_dir, "*.nii.gz")))
        fdg_paths = sorted(glob.glob(os.path.join(self.fdg_dir, "*.nii.gz")))

        def extract_pair_id(path):
            basename = os.path.basename(path).replace(".nii.gz", "")
            parts = basename.split("-")
            subject = parts[0]
            scan_id = int(parts[-1])
            return subject, scan_id

        mri_dict = {extract_pair_id(p): p for p in mri_paths}
        fdg_dict = {extract_pair_id(p): p for p in fdg_paths}
        
        common_keys = sorted(set(mri_dict.keys()) & set(fdg_dict.keys()),
                            key=lambda k: (k[0], k[1]))

        self.pairs = []
        for subject, scan_id in common_keys:
            self.pairs.append((mri_dict[(subject, scan_id)],
                            fdg_dict[(subject, scan_id)],
                            subject, scan_id))  
            
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, str]:
        mri_path, fdg_path, subject, scan_id = self.pairs[index]  

        mri = nifti_to_numpy(mri_path)
        fdg = nifti_to_numpy(fdg_path)
        
        # Augmentation for Classifier Training 
        if self.stage != "train":
            mri = crop(mri)
            fdg = crop(fdg)
        else:
            mri = random_translation(mri)
            fdg = random_translation(fdg)
            
        # Normalization
        mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
        fdg = (fdg - fdg.min()) / (fdg.max() - fdg.min() + 1e-8)

        row = self.df_labels[(self.df_labels["Subject"] == subject) & (self.df_labels["Image_ID"] == scan_id)]
        if row.empty:
            raise ValueError(f"No matching row found for Subject={subject}, Image_ID={scan_id} in CSV.")
        
        label_str = row["Group"].values[0]
        label = torch.tensor(self.group_mapping[label_str], dtype=torch.long)  
        age = torch.tensor(row["Age"].values[0], dtype=torch.float32)

        unique_sample_id = f"{subject}_{scan_id}"

        return mri, fdg, label, age, unique_sample_id

class ThreeClassDataset(Dataset):
    """
    Dataset for MRI-only Classification (DANET Pretraining).
    """
    def __init__(self, root_MRI: str, stage: str = "train", 
                 csv_path: str = "./datas/FDG/MRI/FDG_train_MRI_info.csv"):
        self.mri_dir = root_MRI
        self.stage = stage
        self.df_labels = pd.read_csv(csv_path, encoding="ISO-8859-1")
        self.group_mapping = {"CN": 0, "MCI": 1, "AD": 2}

        mri_paths = sorted(glob.glob(os.path.join(self.mri_dir, "*.nii.gz")))

        def extract_pair_id(path):
            basename = os.path.basename(path).replace(".nii.gz", "")
            parts = basename.split("-")
            subject = parts[0]
            scan_id = int(parts[-1])
            return subject, scan_id

        mri_dict = {extract_pair_id(p): p for p in mri_paths}

        self.samples = []
        for subject, scan_id in sorted(mri_dict.keys(), key=lambda k: (k[0], k[1])):
            self.samples.append((mri_dict[(subject, scan_id)], subject, scan_id))  

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, str]:
        mri_path, subject, scan_id = self.samples[index] 

        mri = nifti_to_numpy(mri_path)
        mri = crop(mri)
        mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)

        row = self.df_labels[(self.df_labels["Subject"] == subject) & (self.df_labels["Image_ID"] == scan_id)]
        if row.empty:
            raise ValueError(f"No matching row found for Subject={subject}, Image_ID={scan_id} in CSV.")
        
        label_str = row["Group"].values[0]
        label = torch.tensor(self.group_mapping[label_str], dtype=torch.long) 
        age = torch.tensor(row["Age"].values[0], dtype=torch.float32)

        unique_sample_id = f"{subject}_{scan_id}"

        return mri, label, age, unique_sample_id
    
