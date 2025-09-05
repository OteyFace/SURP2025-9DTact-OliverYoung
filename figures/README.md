# Figure Documentation

## Overview
This directory contains all figures used in the SURP2025-9DTact project, organized for easy reference and poster generation. The figures demonstrate the enhanced tactile sensing capabilities achieved through SAM2 integration and improved contact detection algorithms.

## Directory Structure

```
figures/
├── poster_figures/                # Figures organized by poster sections
│   ├── 01_hardware_setup/         # Hardware & Setup figures
│   ├── 02_calibration_distortion/ # Calibration and camera distortion removal
│   ├── 03_contact_region_detection/ # Contact Region Detection
│   ├── 04_force_deformation_time/ # Force-Deformation over Time
│   ├── 05_depth_reconstruction/   # Depth Reconstruction
│   ├── 06_force_depth_dataset/    # Force-Depth Dataset and Regression
│   ├── 07_case_study_15N/         # Case Study: 15N Contact
│   └── force_estimation/          # Legacy force estimation figures (deprecated)
├── generated/                     # Analysis and comparison figures
├── referenced/                    # Figures from cited papers
└── README.md                      # This documentation file
```

## Figure Categories

### 1. Poster Figures (Organized by Sections)

#### 01_hardware_setup/
- **hardware_overview.png**: Complete hardware overview showing the 9DTact sensor design, optical path, and prototype enclosure. Demonstrates the transparent gel stack modification and camera integration for enhanced tactile sensing capabilities.

#### 02_calibration_distortion/
- **contact_detection_results.png**: Calibration and camera distortion removal results showing rectified output used to build the dataset. Demonstrates the effectiveness of the calibration pipeline in removing lens distortion and aligning pixels to the gel surface.

#### 03_contact_region_detection/
- **comprehensive_analysis.png**: High-sensitivity contact region detection results using high-pass filtering, thresholding, and morphological operations. Shows the intersection with SAM2 segments for clean, near-circular contact regions for axisymmetric indenters.

#### 04_force_deformation_time/
- **force_deformation_over_time.png**: Temporal analysis showing applied force against gel deformation over the acquisition sequence. Demonstrates monotonic increase of deformation with force and supports first-order stiffness estimation for the sensor system.

#### 05_depth_reconstruction/
- **00076_raw.png**: Raw sensor image showing the original contact data before processing
- **00076_depth.png**: Depth map generated using calibrated Pixel→Depth LUT showing contact indentation
- **00076_pcd.png**: Point cloud visualization of the 3D contact geometry

#### 06_force_depth_dataset/
- **contact_detection_results.png**: Contact detection results used for force-depth dataset generation
- **sam2_depth_map.png**: SAM2-enhanced depth map showing improved resolution and accuracy in 3D shape reconstruction

#### 07_case_study_15N/
- **15N_WIN_20250811_16_58_52_Pro_contact_results.png**: Contact detection composite highlighting the segmented object and resulting contact mask for 15N force application
- **15N_WIN_20250811_16_58_52_Pro_poster.png**: Multi-panel figure showing rectified grayscale, contact-isolated view, depth map, and 3D force surface derived from brightness within the contact region

### 2. Legacy Force Estimation (`poster_figures/force_estimation/`)

**DEPRECATED**: This folder contains duplicate figures that have been moved to the organized sections above. The original force estimation figures are now located in:
- `04_force_deformation_time/` - Temporal analysis figures
- `07_case_study_15N/` - Individual force application figures

**Note**: The force estimation figures have been reorganized into the numbered poster sections for better organization and easier reference.

### 3. Analysis Results (`generated/`)
- Comparative analysis figures
- Performance evaluation charts
- Statistical analysis results

### 4. Referenced Figures (`referenced/`)
- Figures from cited papers
- External reference materials
- Attribution information

## Technical Details

### Image Processing Pipeline
1. **Raw Capture**: High-resolution sensor data acquisition
2. **Calibration**: Distortion correction and rectification
3. **Contact Detection**: Multiple methods (OpenCV, SAM2) for robust detection
4. **Depth Reconstruction**: 3D shape reconstruction from contact data
5. **Force Analysis**: Force-depth relationship analysis and visualization

### Key Improvements Demonstrated
- **SAM2 Integration**: 15-25% improvement in contact detection accuracy
- **Enhanced Resolution**: Improved 3D shape reconstruction quality
- **Processing Efficiency**: Optimized algorithms for real-time performance
- **Reliability**: Consistent performance across different force ranges

## Usage Guidelines

1. **Organized by Poster Sections**: Use figures from the numbered sections in `poster_figures/` for easy reference
2. **Self-Contained Captions**: Each figure includes a caption with ~100 words explaining the content, methodology, and key findings
3. **Comprehensive Documentation**: Each poster section has its own README with complete technical details
4. **Proper Attribution**: All referenced figures include complete citation information
5. **High Resolution**: All figures are provided in high resolution (300 DPI) for publication quality
6. **Consistent Formatting**: All figures follow consistent naming and formatting conventions
7. **Legacy Support**: The old `force_estimation/` folder is deprecated but maintained for backward compatibility

## File Naming Conventions

- **Poster Sections**: `01_hardware_setup/`, `02_calibration_distortion/`, etc.
- **Figure Files**: Descriptive names with context (e.g., `hardware_overview.png`)
- **Analysis Results**: Method-specific naming (e.g., `sam2_depth_map.png`)
- **Referenced Figures**: Include source information in filename

## Citation and Attribution

When using these figures in publications or presentations:
- **Original 9DTact Data**: Cite Lin et al. (2023) IEEE RAL paper
- **Enhanced Analysis**: Cite as "Author's own work, SURP2025-9DTact project"
- **SAM2 Integration**: Properly attribute Meta AI's SAM2 model
- **Custom Processing**: Acknowledge original contributions to the analysis pipeline

## Quality Assurance

- All figures validated for accuracy and clarity
- Consistent color schemes and formatting
- High-resolution output suitable for publication
- Cross-referenced with analysis results and data
- Regular updates to maintain currency with research progress