import os
import torch
from torch.utils.data import Dataset
import nibabel as nib

class BraTSDataset(Dataset):
    def __init__(self, t1_dir, t1gd_dir, transform=None):
        self.t1_paths = sorted([os.path.join(t1_dir, f) for f in os.listdir(t1_dir) if f.endswith('.nii.gz')])
        self.t1gd_paths = sorted([os.path.join(t1gd_dir, f) for f in os.listdir(t1gd_dir) if f.endswith('.nii.gz')])
        self.transform = transform

    def __len__(self):
        return len(self.t1_paths)

    def __getitem__(self, idx):
        t1_img = nib.load(self.t1_paths[idx]).get_fdata()
        t1gd_img = nib.load(self.t1gd_paths[idx]).get_fdata()

        if self.transform:
            t1_img, t1gd_img = self.transform(t1_img, t1gd_img)

        t1_tensor = torch.tensor(t1_img, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        t1gd_tensor = torch.tensor(t1gd_img, dtype=torch.float32).unsqueeze(0)
        return t1_tensor, t1gd_tensor
