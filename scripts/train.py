import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.data_loader import BraTSDataset
from utils.metrics import compute_psnr, compute_ssim

# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths and hyperparameters from config
PREPROCESSED_TRAIN_DIR = config['data']['preprocessed_train_dir']
PREPROCESSED_VAL_DIR = config['data']['preprocessed_val_dir']
BATCH_SIZE = config['training']['batch_size']
NUM_EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
SAVE_BEST_MODEL = config['training']['save_best_model']

# Datasets
train_dataset = BraTSDataset(
    t1_dir=os.path.join(PREPROCESSED_TRAIN_DIR, config['data']['t1_subdir']),
    t1gd_dir=os.path.join(PREPROCESSED_TRAIN_DIR, config['data']['t1gd_subdir'])
)
val_dataset = BraTSDataset(
    t1_dir=os.path.join(PREPROCESSED_VAL_DIR, config['data']['t1_subdir']),
    t1gd_dir=os.path.join(PREPROCESSED_VAL_DIR, config['data']['t1gd_subdir'])
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model, Loss, Optimizer
model = UNet(in_channels=1, out_channels=1).cuda()
loss_fn = nn.L1Loss() if config['training']['loss_function'] == "l1_loss" else nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    # Training and validation logic...
    if SAVE_BEST_MODEL and val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(config['checkpoints_dir'], "best_model.pth"))
