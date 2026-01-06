import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import sys

def split_atlas_to_masks(atlas_path: str, out_dir: str):
    """
    Split a multi-label NIfTI atlas into individual binary mask files.
    
    Args:
        atlas_path: Path to the input atlas file (.nii or .nii.gz). 
        out_dir: Directory to save the generated binary masks.
    """
    
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Info] Output directory: {out_dir}")

    print(f"[Info] Loading atlas from: {atlas_path}")
    atlas_img = nib.load(atlas_path)

    atlas_data = atlas_img.get_fdata().astype(np.int32)
    affine = atlas_img.affine
    header = atlas_img.header

    labels = np.unique(atlas_data)
    labels = np.sort(labels)
    
    print(f"[Info] Found {len(labels)} unique labels (including background).")
    print(f"[Info] Labels: {labels}")

    for lbl in tqdm(labels, desc="Splitting masks"):
        mask_data = (atlas_data == lbl).astype(np.uint8)
        
        mask_img = nib.Nifti1Image(mask_data, affine, header)
        
        out_fn = os.path.join(out_dir, f'{lbl:03d}.nii.gz')
        
        nib.save(mask_img, out_fn)

    print(f"[Success] All {len(labels)} masks have been saved to {out_dir}")

if __name__ == "__main__":
    """
    Usage Example:
    python tools/split_atlas.py --input ./datas/HarvardOxford-sub-maxprob-thr0-1mm_aligned.nii.gz --output ./datas/atlas_masks/
    """
    parser = argparse.ArgumentParser(description="Split 3D Atlas into Binary Masks ")
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to the input NIfTI atlas file.')
    parser.add_argument('--output', type=str, default='./datas/atlas_masks', 
                        help='Directory to save the output masks.')
    
    args = parser.parse_args()
    
    try:
        split_atlas_to_masks(args.input, args.output)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)