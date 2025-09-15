from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

def calculate_ssim(img1_path, img2_path, read_image=True):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Parameters:
        image1 (ndarray): The first input image.
        image2 (ndarray): The second input image.

    Returns:
        float: The SSIM value between the two images.
    """
    if not read_image:
        image1 = img1_path
        image2 = img2_path
    else:
        image1 = cv2.imread(img1_path)
        image2 = cv2.imread(img2_path) 

    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images if dimensions don't match
    if image1.shape != image2.shape:
        height, width = image1.shape
        image2 = cv2.resize(image2, (width, height), interpolation=cv2.INTER_AREA)

    # Compute SSIM
    ssim_value, _ = ssim(image1, image2, full=True)
    return ssim_value

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python ssim_metric.py <image1> <image2>")
        sys.exit(1)
        
    image1 = sys.argv[1]
    image2 = sys.argv[2]

    if image1 is None or image2 is None:
        print("Error: Could not load images.")
    else:
        ssim_value = calculate_ssim(image1, image2)
        print(f"SSIM: {ssim_value:.4f}")
