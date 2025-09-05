# Calibration and Camera Distortion Removal

## Overview
This section contains figures demonstrating the calibration process and camera distortion removal capabilities of the enhanced 9DTact system. The figures show the rectification process and its effectiveness in preparing data for contact detection and depth reconstruction.

## Figures

### contact_detection_results.png
**Caption**: Calibration and camera distortion removal results showing the rectified output used to build the dataset. The figure demonstrates the effectiveness of the calibration pipeline in removing lens distortion and aligning pixels to the gel surface before any masking or depth conversion operations.

**Technical Details**:
- **Process**: Rectification aligns pixels to the gel and removes lens distortion
- **Method**: Leverages SAM2 on RGB frames to separate foreground from background
- **Compatibility**: Maintains compatibility with original 9DTact calibration flow
- **Output**: Rectified images ready for contact detection and depth mapping

**Key Features**:
- **Distortion Correction**: Removes barrel/pincushion distortion from camera lens
- **Pixel Alignment**: Aligns image pixels to the gel surface geometry
- **Background Separation**: Uses SAM2 for intelligent foreground/background separation
- **Data Preparation**: Prepares clean data for subsequent processing steps

**Research Context**: This calibration step is crucial for accurate contact detection and depth reconstruction. The rectified output provides the foundation for all subsequent analysis, ensuring that contact regions are properly identified and depth measurements are accurate.
