import argparse
import ssim_metric
import psnr_metric
import lpips_metric

# compute image metrics for a single image pair
def image_metrics(image_path,             # test image
                  gt_image_path):         # ground truth image
    ssim_score = ssim_metric.calculate_ssim(gt_image_path, image_path)
    psnr_score = psnr_metric.calculate_psnr(gt_image_path, image_path)
    lpips_score = lpips_metric.calculate_lpips(gt_image_path, image_path)
    return ssim_score, psnr_score, lpips_score

#-----

# main

# parse arguments
parser = argparse.ArgumentParser(description="single-image-pair.py </path/to/image.png> </path/to/ground_truth_image.png>")
parser.add_argument("image_path", help="path/filename of image (in .png format)")
parser.add_argument("gt_image_path", help="path/filename of ground truth image (in .png format)")
args = parser.parse_args()

# compute metrics
ssim_score, psnr_score, lpips_score = image_metrics(args.image_path, args.gt_image_path)

# print results
print("SSIM", ssim_score, "PSNR", psnr_score, "LPIPS", lpips_score)
