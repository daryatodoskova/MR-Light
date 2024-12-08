import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_slice(image, title=None, cmap='gray', save_path=None):
    """
    Visualizes a single slice of a 3D volume.

    Parameters:
    - image (numpy.ndarray or torch.Tensor): The 2D slice to display.
    - title (str): Title of the plot.
    - cmap (str): Colormap to use.
    - save_path (str): If provided, saves the plot to this path.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_comparison(t1_slice, generated_slice, ground_truth_slice, save_path=None):
    """
    Visualizes side-by-side comparison of T1, generated T1Gd, and ground truth T1Gd.

    Parameters:
    - t1_slice (numpy.ndarray or torch.Tensor): The input T1 slice.
    - generated_slice (numpy.ndarray or torch.Tensor): The generated T1Gd slice.
    - ground_truth_slice (numpy.ndarray or torch.Tensor): The ground truth T1Gd slice.
    - save_path (str): If provided, saves the plot to this path.
    """
    if isinstance(t1_slice, torch.Tensor):
        t1_slice = t1_slice.cpu().numpy()
    if isinstance(generated_slice, torch.Tensor):
        generated_slice = generated_slice.cpu().numpy()
    if isinstance(ground_truth_slice, torch.Tensor):
        ground_truth_slice = ground_truth_slice.cpu().numpy()
    
    plt.figure(figsize=(18, 6))
    titles = ['Input T1', 'Generated T1Gd', 'Ground Truth T1Gd']
    slices = [t1_slice, generated_slice, ground_truth_slice]
    
    for i, (img, title) in enumerate(zip(slices, titles)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_training_progress(losses, psnr_scores, ssim_scores, save_path=None):
    """
    Plots the training progress over epochs.

    Parameters:
    - losses (list of float): Training loss per epoch.
    - psnr_scores (list of float): PSNR values per epoch.
    - ssim_scores (list of float): SSIM values per epoch.
    - save_path (str): If provided, saves the plot to this path.
    """
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid()
    plt.legend()
    
    # Metrics subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, psnr_scores, label='PSNR', color='green')
    plt.plot(epochs, ssim_scores, label='SSIM', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Metrics Over Time')
    plt.grid()
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_images_as_grid(image_list, titles, save_path):
    """
    Saves multiple images as a grid.

    Parameters:
    - image_list (list of numpy.ndarray or torch.Tensor): List of 2D slices to display.
    - titles (list of str): Titles for each image.
    - save_path (str): Path to save the grid.
    """
    num_images = len(image_list)
    plt.figure(figsize=(4 * num_images, 4))
    
    for i, (image, title) in enumerate(zip(image_list, titles)):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()
