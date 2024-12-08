import os
import yaml
from utils.preprocessing import preprocess_data

# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths from config
TRAIN_DIR = config['data']['training_dir']
VAL_DIR = config['data']['validation_dir']
PREPROCESSED_TRAIN_DIR = config['data']['preprocessed_train_dir']
PREPROCESSED_VAL_DIR = config['data']['preprocessed_val_dir']
TARGET_SHAPE = tuple(config['preprocessing']['target_shape'])

# Preprocess training data
print("Preprocessing training data...")
preprocess_data(input_dir=TRAIN_DIR, output_dir=PREPROCESSED_TRAIN_DIR, target_shape=TARGET_SHAPE)

# Preprocess validation data
print("Preprocessing validation data...")
preprocess_data(input_dir=VAL_DIR, output_dir=PREPROCESSED_VAL_DIR, target_shape=TARGET_SHAPE)

print("Preprocessing completed.")
