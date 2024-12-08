import os
import yaml
import torch
from models.unet import UNet
from utils.data_loader import BraTSDataset
from utils.metrics import compute_psnr, compute_ssim

# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths
PREPROCESSED_VAL_DIR = config['data']['preprocessed_val_dir']
CHECKPOINT_PATH = os.path.join(config['checkpoints_dir'], "best_model.pth")

# Dataset and DataLoader
val_dataset = BraTSDataset(
    t1_dir=os.path.join(PREPROCESSED_VAL_DIR, config['data']['t1_subdir']),
    t1gd_dir=os.path.join(PREPROCESSED_VAL_DIR, config['data']['t1gd_subdir'])
)
val_loader = DataLoader(val_dataset, batch_size=1)

# Load Model
model = UNet(in_channels=1, out_channels=1).cuda()
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# Evaluation
psnr_list, ssim_list = [], []
for t1, t1gd in val_loader:
    t1, t1gd = t1.cuda(), t1gd.cuda()
    with torch.no_grad():
        output = model(t1)
    psnr_list.append(compute_psnr(output, t1gd))
    ssim_list.append(compute_ssim(output, t1gd))

print(f"Validation PSNR: {sum(psnr_list) / len(psnr_list):.2f}, SSIM: {sum(ssim_list) / len(ssim_list):.4f}")
