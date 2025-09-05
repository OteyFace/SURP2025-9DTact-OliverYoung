# Depth Reconstruction

## Overview
This section contains figures demonstrating the 3D depth reconstruction capabilities of the enhanced 9DTact sensor system. The figures show the complete pipeline from raw sensor data to 3D point cloud visualization, highlighting the accuracy and detail of the depth reconstruction process.

## Figures

### 00076_raw.png
**Caption**: Raw sensor image showing the original contact data before processing. This figure demonstrates the initial sensor response to object contact, providing the foundation for all subsequent processing steps including contact detection, depth mapping, and 3D reconstruction.

**Technical Details**:
- **Source**: Direct sensor output before any processing
- **Format**: High-resolution RGB image from transparent gel sensor
- **Content**: Contact deformation pattern visible in gel surface
- **Quality**: Unprocessed data showing natural sensor response

### 00076_depth.png
**Caption**: Depth map generated using calibrated Pixel→Depth LUT showing contact indentation. The figure demonstrates the conversion from pixel intensity to physical depth measurements, providing quantitative 3D information about the contact geometry and deformation magnitude.

**Technical Details**:
- **Method**: Calibrated Pixel→Depth LUT (Pixel_to_Depth_iterative.npy)
- **Process**: Piecewise linear mapping gated by lighting threshold
- **Output**: Depth values in millimeters
- **Smoothing**: Small Gaussian kernels to reduce quantization noise

**Key Features**:
- **Calibration**: Built from shallow-to-deep calibration video
- **Accuracy**: Precise depth measurements in physical units
- **Resolution**: High-resolution depth mapping
- **Noise Reduction**: Gaussian smoothing while preserving shape

### 00076_pcd.png
**Caption**: Point cloud visualization of the 3D contact geometry. This figure demonstrates the final 3D reconstruction output, showing the complete spatial distribution of contact forces and providing detailed geometric information about the contact interface.

**Technical Details**:
- **Generation**: Converted from depth map to 3D point cloud
- **Visualization**: Color-coded by depth/elevation
- **Geometry**: Complete 3D surface representation
- **Analysis**: Enables spatial force distribution analysis

**Research Context**: These figures demonstrate the complete depth reconstruction pipeline from raw sensor data to 3D visualization. The process enables accurate measurement of contact geometry, force distribution, and object shape reconstruction, supporting applications in robotics, haptics, and tactile sensing research.
