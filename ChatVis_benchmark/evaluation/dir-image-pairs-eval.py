from pathlib import Path
import argparse
import ssim_metric
import psnr_metric
import lpips_metric

# compute average image metrics for a pair of image directories
def average_metrics(test_dir,           # directory of test images
                    gt_dir):            # directory of ground truth images
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    for img_path in test_dir.glob('*.png'):
        img_name = img_path.name
        gt_name = img_name.replace('.png', '-gt.png')
        gt_img_path = gt_dir / gt_name
        if gt_img_path.exists():
            ssim_score = ssim_metric.calculate_ssim(gt_img_path, img_path)
            print("ssim_score", ssim_score)
            ssim_scores.append(ssim_score)
            psnr_score = psnr_metric.calculate_psnr(gt_img_path, img_path)
            print("psnr_score", psnr_score)
            psnr_scores.append(psnr_score)
            lpips_score = lpips_metric.calculate_lpips(gt_img_path, img_path)
            print("lpips_score", lpips_score)
            lpips_scores.append(lpips_score)
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0
    avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else 0
    return avg_ssim, avg_psnr, avg_lpips

#-----

# main

# parse arguments
parser = argparse.ArgumentParser(description="dir-image-pairs.py </path/to/image_directory> </path/to/ground_truth_image_directory>")
parser.add_argument("image_path", help="path of image directory")
parser.add_argument("gt_image_path", help="path of ground truth image directory")
args = parser.parse_args()

# compute metrics
avg_ssim, avg_psnr, avg_lpips = average_metrics(Path(args.image_path), Path(args.gt_image_path))

# print results
print("average SSIM", avg_ssim, "average PSNR", avg_psnr, "average LPIPS", avg_lpips)

