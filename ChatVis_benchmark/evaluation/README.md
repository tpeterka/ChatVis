# ChatVis_benchmark evaluation scripts

One can evaluate generated images against ground truth images using several image metrics such as PSNR, SSIM, and LPIPS.

## Setup

It is recommended to install dependencies in a virtual Python environment or a Conda environment.

Virtual Python environment:
```
python -m venv .venv
source .venv/bin/activate
pip3 install <dependencies>     # first time only
cd /path/to/ChatVis_benchmark/evaluation
python3 <evaluation_script>
```

Conda environment (preferred):
```
conda activate
pip3 install <dependencies>     # first time only
cd /path/to/ChatVis_benchmark/evaluation
python3 <evaluation_script>
conda deactivate
```

Install dependencies in your environment:
```
pip3 install numpy
pip3 install scikit-image
pip3 install opencv-python
pip3 install lpips
pip3 install tabulate
```

## Execution

Image metrics for a single image pair (in `.png` format)
```
cd /path/to/ChatVis_benchmark/evaluation
python3 single-image-pair-eval.py <image.png> <ground_truth_image.png>
```

Image metrics averaged over a pair of directories of images (in `.png` format)

```
cd /path/to/ChatVis_benchmark/evaluation
python3 dir-image-pairs-eval.py <images_directory> <ground_truth_images_directory>
```

The image file names are expected to have the same base names, with the ground truth images having "-gt" suffix. E.g.:
```
> ls images_directory
image1.png
image2.png
image3.png

> ls ground_truth_images_directory
image1-gt.png
image2-gt.png
image3-gt.png
```
