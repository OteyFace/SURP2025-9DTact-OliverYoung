# Force-Depth Dataset and Regression

## Overview
This section contains figures demonstrating the force-depth dataset generation and regression analysis capabilities of the enhanced 9DTact sensor system. The figures show the relationship between contact depth and applied force, supporting the development of accurate force estimation models.

## Figures

### contact_detection_results.png
**Caption**: Contact detection results used for force-depth dataset generation. This figure shows the rectified and processed contact regions that serve as the foundation for extracting depth statistics and building the force-depth relationship dataset used in regression analysis.

**Technical Details**:
- **Source**: Rectified contact detection output
- **Processing**: Background removal and contact region isolation
- **Statistics**: Mean, max, and sum depth calculations within contact regions
- **Dataset**: CSV format with filename, force_N, mean_depth_mm, max_depth_mm, sum_depth_mm

**Key Features**:
- **Rectification**: Distortion-corrected contact regions
- **Masking**: Clean contact region boundaries
- **Statistics**: Quantitative depth measurements
- **Data Format**: Structured for machine learning applications

### sam2_depth_map.png
**Caption**: SAM2-enhanced depth map showing improved resolution and accuracy in 3D shape reconstruction. This figure demonstrates the superior performance of SAM2 integration in generating high-quality depth maps for force-depth relationship analysis and regression model development.

**Technical Details**:
- **Method**: SAM2-enhanced depth reconstruction
- **Quality**: Improved resolution and accuracy
- **Processing**: Enhanced contact region detection
- **Output**: High-quality depth maps for analysis

**Performance Improvements**:
- **Resolution**: Higher detail in depth reconstruction
- **Accuracy**: More precise depth measurements
- **Noise Reduction**: Better suppression of artifacts
- **Consistency**: Reliable performance across different objects

**Research Context**: These figures demonstrate the dataset generation process for force estimation models. The combination of accurate contact detection and high-quality depth reconstruction enables the development of robust regression models that can predict applied forces from sensor measurements, supporting applications in robotics and haptic feedback systems.
