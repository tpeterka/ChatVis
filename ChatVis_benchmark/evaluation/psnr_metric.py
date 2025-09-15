import numpy as np
import cv2

def calculate_psnr(img1_path, img2_path, read_image=True):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        original (ndarray): Original image.
        compressed (ndarray): Compressed image.

    Returns:
        float: PSNR value in decibels.
    """

    if not read_image:
        original = img1_path
        compressed = img2_path
    else:
        original = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        compressed = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE) 

    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(compressed.shape) == 3:
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

    # Resize images if dimensions don't match
    if original.shape != compressed.shape:
        height, width = original.shape
        compressed = cv2.resize(compressed, (width, height), interpolation=cv2.INTER_AREA)
         
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    max_pixel = 255.0  # Assuming the image is in 8-bit format
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

if __name__ == "__main__":
    image1 = sys.argv[1]
    image2 = sys.argv[2]

    if image1 is None or image2 is None:
        print("Error: Could not load images.")
    else:
        psnr_value = calculate_psnr(image1, image2)
        print(f"PSNR: {psnr_value:.2f} dB")
