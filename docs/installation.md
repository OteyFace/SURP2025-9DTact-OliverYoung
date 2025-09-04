# Installation Guide

## Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

## Setup Instructions

### 1. Clone the Repository
```bash
git clone [your-repository-url]
cd SURP2025-9DTact
```

### 2. Create Conda Environment
```bash
conda create -n 9dtact python=3.8
conda activate 9dtact
```

### 3. Install PyTorch
```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 4. Install Additional Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install SAM2 (for enhanced segmentation)
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

## Verification
Run the following to verify installation:
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## Troubleshooting
- If you encounter CUDA issues, ensure your GPU drivers are up to date
- For SAM2 installation issues, check the official SAM2 repository
- Contact [your-email] for additional support
