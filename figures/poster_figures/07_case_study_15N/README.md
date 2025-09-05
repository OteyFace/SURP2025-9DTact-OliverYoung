# Case Study: 15N Contact

## Overview
This section contains figures demonstrating a comprehensive case study of 15N force application on the enhanced 9DTact sensor system. The figures show the complete pipeline from contact detection to 3D force visualization, highlighting the sensor's capabilities in high-force applications.

## Figures

### 15N_WIN_20250811_16_58_52_Pro_contact_results.png
**Caption**: Contact detection composite highlighting the segmented object and the resulting contact mask for 15N force application. This figure demonstrates the enhanced contact detection capabilities using high-pass filtering, thresholding, and morphological operations, intersected with SAM2 segments for clean contact region identification.

**Technical Details**:
- **Force Level**: 15N applied force
- **Method**: High-pass filtering + threshold + morphology + SAM2 intersection
- **Output**: Clean contact mask with suppressed background
- **Quality**: Near-circular contact region for axisymmetric indenter

**Key Features**:
- **Contact Mask**: High-pass filtering + threshold + morphology
- **SAM2 Integration**: Intersected with SAM2 segments for background suppression
- **Threshold Tuning**: Optimized for circular objects
- **Noise Reduction**: Clean contact boundaries

### 15N_WIN_20250811_16_58_52_Pro_poster.png
**Caption**: Multi-panel figure showing rectified grayscale, contact-isolated view, depth map, and 3D force surface derived from brightness within the contact region and normalized to the labeled force. This comprehensive visualization demonstrates the complete sensor pipeline from raw data to force estimation.

**Technical Details**:
- **Panels**: Four-panel comprehensive analysis
- **Content**: Rectified grayscale, isolated contact, depth map, 3D force surface
- **Force Surface**: Per-pixel force F(x,y) computed by normalizing grayscale within mask
- **Scaling**: Normalized to labeled force in newtons

**Key Components**:
- **Rectified Grayscale**: Distortion-corrected sensor output
- **Contact Isolation**: Clean contact region extraction
- **Depth Map**: 3D contact geometry visualization
- **Force Surface**: Per-pixel force distribution

**Interpretation Notes**:
- **Contact Mask**: High-pass filtering + threshold + morphology, intersected with SAM2 segments
- **Depth Map**: Derived via calibrated Pixel→Depth LUT; brighter hues = larger indentation depths
- **3D Force Surface**: Per-pixel force F(x,y) computed by normalizing grayscale within mask to [0,1] and scaling by labeled force; peaks indicate higher local pressure

**Research Context**: This case study demonstrates the sensor's performance under high-force conditions (15N), showing robust contact detection, accurate depth reconstruction, and detailed force visualization. The comprehensive analysis validates the sensor's capabilities for applications requiring high-force tactile sensing and provides insights into the spatial distribution of contact forces.
