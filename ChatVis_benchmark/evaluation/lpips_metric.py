import lpips
import torch
import cv2
from PIL import Image
import numpy as np

def calculate_lpips(image1_path, image2_path, net='alex', read_image=True):
    """
    Calculate the Learned Perceptual Image Patch Similarity (LPIPS) between two images.

    Parameters:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        net (str): Network to use for perceptual similarity ('alex', 'vgg', 'squeeze').

    Returns:
        float: LPIPS distance between the two images.
    """
    # Load LPIPS model
    loss_fn = lpips.LPIPS(net=net)  # Choose 'alex', 'vgg', or 'squeeze'
    
    # Load images
    if not read_image:
        img1 = image1_path
        img2 = image2_path
    else:
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path) 
    # Check dimensions and resize if necessary
    if img1.shape[:2] != img2.shape[:2]:
        img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_AREA)

    img1 = (img1.transpose(2, 0, 1).astype(np.float32) / 255) * 2 - 1
    img2 = (img2.transpose(2, 0, 1).astype(np.float32) / 255) * 2 - 1

    if img1 is None or img2 is None:
        raise ValueError("Error loading images.")


    # Add batch dimension
    img1_tensor = torch.from_numpy(img1).unsqueeze(0)
    img2_tensor = torch.from_numpy(img2).unsqueeze(0)

    # Calculate LPIPS distance
    distance = loss_fn(img1_tensor, img2_tensor)
    return distance.item()

if __name__ == "__main__":
    image1 = "image1.png"
    image2 = "image2.png"
    lpips_value = calculate_lpips(image1, image2)
    print(f"LPIPS distance: {lpips_value:.4f}")
