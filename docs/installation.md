# Installation Guide

## Prerequisites
- Python 3.8+
- Git
- (Recommended) NVIDIA GPU with CUDA

## Quickstart (Conda)
```bash
# 1) Clone
git clone https://github.com/OteyFace/SURP2025-9DTact.git
cd SURP2025-9DTact

# 2) Create & activate environment
conda create -n 9dtact python=3.8 -y
conda activate 9dtact

# 3) Install PyTorch
#    GPU (CUDA 11.8):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
#    OR CPU-only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 4) Project dependencies
python -m pip install -U pip
pip install -r requirements.txt

# 5) (If SAM2 isn't in requirements.txt) install Segment Anything 2
#    Comment this out if requirements already provides it.
# pip install git+https://github.com/facebookresearch/segment-anything-2.git

Verify
python -c "import torch, cv2; print('PyTorch:', torch.__version__, '| OpenCV:', cv2.__version__)"

Notes

Linux may require GUI/backends for OpenCV:

sudo apt-get update && sudo apt-get install -y libgl1 ffmpeg


On Windows, run these commands from an Anaconda/Miniconda PowerShell.

Troubleshooting

CUDA errors → update NVIDIA drivers and ensure CUDA 11.8-compatible build is installed.

SAM2 build issues → check the Segment Anything 2 repository instructions.

Still stuck? Email: ojyoung@ucsc.edu