from tabulate import tabulate
import sambanova_ai4s as sn
import argo_ai4s as argo
import argo_o1preview_ai4s as argo_o1

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import ssim_metric
import psnr_metric
import lpips_metric


def remove_background(image):
    import cv2
    from sklearn.cluster import KMeans

    # Reshape the image into a 2D array of pixels (ignoring alpha if present)
    pixels = image[:, :, :3].reshape(-1, 3)

    # Use KMeans clustering to find the most dominant color (background)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    # Create a mask for pixels close to the dominant color
    lower_bound = np.clip(dominant_color - 30, 0, 255).astype(np.uint8)
    upper_bound = np.clip(dominant_color + 30, 0, 255).astype(np.uint8)
    mask = cv2.inRange(image[:, :, :3], lower_bound, upper_bound)

    # Invert the mask to keep the foreground
    mask_inv = cv2.bitwise_not(mask)

    # Extract the foreground
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)

    # Find bounding box of the foreground
    coords = cv2.findNonZero(mask_inv)  # Find non-zero points (foreground)
    x, y, w, h = cv2.boundingRect(coords)  # Get bounding box
    cropped_foreground = foreground[y:y+h, x:x+w]

    return cropped_foreground

#TODO: For video, we would need to modify/extend this function
def calculate_metrics(model_dir, full_paper_dir):
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    ssim_scores_preprocessed = []
    psnr_scores_preprocessed = []
    lpips_scores_preprocessed = []
    for img_path in model_dir.glob('*.png'):
        img_name = img_path.name
        subfolder = '-'.join(img_name.split('-')[:-1]) if 'screenshot' in img_name else '-'.join(img_name.split('-')[:-1])
        gt_name = img_name.replace('.png', '-gt.png')
        gt_img_path = full_paper_dir / subfolder / gt_name
        if gt_img_path.exists():
            ssim_score = ssim_metric.calculate_ssim(gt_img_path, img_path)
            ssim_scores.append(ssim_score)
            psnr_score = psnr_metric.calculate_psnr(gt_img_path, img_path)
            psnr_scores.append(psnr_score)
            lpips_score = lpips_metric.calculate_lpips(gt_img_path, img_path)
            lpips_scores.append(lpips_score)
            # Preprocess the image. For now, calculating image quality scores both with original and preprocessed images, probably will keep only one depending on the results.
            #TODO: If not much difference between original and preprocessed image quality scores, keep the original one. 
            img_path_preprocessed = remove_background(cv2.imread(img_path))
            gt_img_path_preprocessed = remove_background(cv2.imread(gt_img_path))
            ssim_score_preprocessed = ssim_metric.calculate_ssim(gt_img_path_preprocessed, img_path_preprocessed, False)
            ssim_scores_preprocessed.append(ssim_score_preprocessed)
            psnr_score_preprocessed = psnr_metric.calculate_psnr(gt_img_path_preprocessed, img_path_preprocessed, False)
            psnr_scores_preprocessed.append(psnr_score_preprocessed)
            lpips_score_preprocessed = lpips_metric.calculate_lpips(gt_img_path_preprocessed, img_path_preprocessed, 'alex', False)
            lpips_scores_preprocessed.append(lpips_score_preprocessed)
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0
    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    avg_ssim_preprocessed = sum(ssim_scores_preprocessed) / len(ssim_scores_preprocessed) if ssim_scores_preprocessed else 0
    avg_psnr_preprocessed = sum(psnr_scores_preprocessed) / len(psnr_scores_preprocessed) if psnr_scores_preprocessed else 0
    avg_lpips_preprocessed = sum(lpips_scores_preprocessed) / len(lpips_scores_preprocessed) if lpips_scores_preprocessed else 0
    return avg_ssim, avg_psnr, avg_lpips, avg_ssim_preprocessed, avg_psnr_preprocessed, avg_lpips_preprocessed

# Get current working directory and parent
cwd = Path(os.getcwd())
parent = cwd.parent

# Create full-paper directory
full_paper_dir = parent / 'examples' / 'full-paper'
full_paper_dir.mkdir(parents=True, exist_ok=True)

# Source directories to copy from
source_dirs = [
    #parent / 'examples' / 'AI4S-paper',
    #parent / 'examples' / 'finished-regression-tests',
    #parent / 'examples' / 'o1-preview',
    parent / 'examples' / 'science-cases' 
]

# Copy contents of each source directory to full-paper
for src_dir in source_dirs:
    if src_dir.exists():
        for item in src_dir.glob('*'):
            if item.is_dir():
                shutil.copytree(item, full_paper_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, full_paper_dir / item.name)

data = []
argo_models = ["gpt35", "gpt4o"] 
sambanova_models = ["DeepSeek-R1-Distill-Llama-70B", "Meta-Llama-3.1-405B-Instruct"] #["Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.1-70B-Instruct", "Meta-Llama-3.1-405B-Instruct", "Meta-Llama-3.3-70B-Instruct", "QwQ-32B-Preview", "Qwen2.5-Coder-32B-Instruct", "Qwen2.5-72B-Instruct"]

#TODO: incorporate the image metrics for SambaNova models as in OpenAI models below
#for snm in sambanova_models:
#    success = sn.eval_model(snm)
#    data.append([snm, success*100])

for am in argo_models:
    model_dir = Path(str(cwd)) / am
    if model_dir.exists():
        shutil.rmtree(model_dir) 
    success = argo.eval_model(am)
    avg_ssim, avg_psnr, avg_lpips, avg_ssim_preprocessed, avg_psnr_preprocessed, avg_lpips_preprocessed = calculate_metrics(model_dir, full_paper_dir) 
    data.append([am, success*100, avg_ssim*success, avg_psnr*success, avg_lpips*success, avg_ssim_preprocessed*success, avg_psnr_preprocessed*success, avg_lpips_preprocessed*success])

model_name = "gpto1preview"
model_dir = Path(str(cwd)) / model_name
if model_dir.exists():
    shutil.rmtree(model_dir)
success = argo_o1.eval_model("gpto1preview")
avg_ssim, avg_psnr, avg_lpips, avg_ssim_preprocessed, avg_psnr_preprocessed, avg_lpips_preprocessed = calculate_metrics(model_dir, full_paper_dir)

data.append(["gpto1preview", success*100, avg_ssim*success, avg_psnr*success, avg_lpips*success, avg_ssim_preprocessed*success, avg_psnr_preprocessed*success, avg_lpips_preprocessed*success])

headers = ["Model", "CompletionRate", "SSIM", "PSNR", "LPIPS", "Preprocessed SSIM", "Preprocessed PSNR", "Preprocessed LPIPS"]

table_str = tabulate(data, headers=headers, tablefmt="github")
print(table_str)

output_file = "argo_fullPaper_output.md" #"sambanova_ai4s_output.md" #argo_ai4s_output.md
#output_file = "ai4s_output.md"
with open(output_file, "w") as file:
    file.write(table_str)


