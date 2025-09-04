# Methodology

## Project Overview
This project enhances the 9DTact tactile sensor system with improved contact analysis and shape reconstruction capabilities.

## My Contributions

### 1. Contact Accuracy Analysis
- **File**: `code/my_contributions/contact_accuracy_analysis.py`
- **Purpose**: Quantitative analysis of contact detection accuracy
- **Method**: SAM2-based segmentation with OpenCV comparison
- **Metrics**: Contact diameter, area, circularity, and shape analysis

### 2. Enhanced Shape Reconstruction
- **Files**: `code/my_contributions/New_*.py`
- **Improvements**:
  - SAM2 integration for better contact detection
  - Enhanced camera calibration
  - Improved sensor calibration
  - Better shape reconstruction algorithms

### 3. SAM2 Integration
- **Purpose**: Modern computer vision for tactile sensing
- **Benefits**: More accurate segmentation and contact detection
- **Implementation**: Custom notebooks and examples

## Dependencies
- Original 9DTact code in `code/original_9dtact/`
- SAM2 for enhanced segmentation
- OpenCV for image processing
- PyTorch for deep learning components

## Usage Examples

### Contact Accuracy Analysis
```python
python code/my_contributions/contact_accuracy_analysis.py \
  --benchmark data/analysis_results/benchmark.png \
  --candidate_left data/analysis_results/candidate1.png \
  --candidate_right data/analysis_results/candidate2.png \
  --output_dir data/analysis_results/
```

### Enhanced Shape Reconstruction
```python
cd code/my_contributions/
python New_Shape_Reconstruction.py
```

## Results
See `docs/results.md` for detailed analysis results and performance metrics.
