import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def load_nifti(file_path):
    """Load a NIfTI file and return the data array."""
    img = nib.load(file_path)
    return img.get_fdata()

def save_nifti(data, reference_file, output_path):
    """Save a NIfTI file with reference header and affine."""
    ref_img = nib.load(reference_file)
    nib.save(nib.Nifti1Image(data, ref_img.affine, ref_img.header), output_path)

def normalize_intensity(data):
    """Normalize intensity to zero mean and unit variance."""
    return (data - np.mean(data)) / np.std(data)

def resample_image(data, target_shape):
    """Resample image to the target shape using zoom."""
    factors = [t / s for t, s in zip(target_shape, data.shape)]
    return zoom(data, factors, order=3)  # Cubic interpolation

def preprocess_data(input_dir, output_dir, target_shape=(240, 240, 155)):
    """Preprocess all NIfTI files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.nii.gz'):
            file_path = os.path.join(input_dir, file_name)
            data = load_nifti(file_path)
            data = normalize_intensity(data)
            data = resample_image(data, target_shape)
            save_nifti(data, file_path, os.path.join(output_dir, file_name))
            print(f"Processed {file_name}")
