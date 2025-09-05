# Force Estimation Visualization Figures

## ⚠️ DEPRECATED - This folder is maintained for backward compatibility only

**NEW LOCATION**: These figures have been reorganized into the numbered poster sections for better organization:

- **Temporal Analysis**: `04_force_deformation_time/force_deformation_over_time.png`
- **15N Case Study**: `07_case_study_15N/15N_WIN_20250811_16_58_52_Pro_poster.png`
- **Other Force Applications**: See `07_case_study_15N/` for individual force application figures

## Overview
This directory contains 3D force visualization figures generated for the 9DTact sensor system. Each figure demonstrates the sensor's response to different applied forces, showing both magnitude and spatial distribution of contact pressure.

## Figure Categories

### Temporal Analysis
- **force_deformation_over_time.png**: Shows the relationship between applied force and gel deformation over time, demonstrating the sensor's ability to track force changes and estimate material stiffness.

### Individual Force Applications
- **15N_WIN_20250811_16_58_52_Pro_poster.png**: Comprehensive 3D force visualization for 15N force application, showing contact detection, depth reconstruction, and force surface generation.

## Technical Details

### Force Visualization Method
- **Per-pixel Force Calculation**: F(x,y) = (normalized_grayscale) × (labeled_force)
- **Normalization**: Grayscale values within contact mask normalized to [0,1] range
- **Scaling**: Multiplied by ground truth force measurement in newtons
- **Visualization**: 3D surface plot with color-coded force magnitude

### Key Features
- **Spatial Distribution**: Shows how force is distributed across the contact area
- **Peak Detection**: Identifies areas of highest local pressure
- **Force Magnitude**: Quantitative force measurements in newtons
- **Contact Geometry**: Relationship between contact shape and force distribution

## Usage Notes

**Please use the new organized structure**:
- For temporal analysis: `04_force_deformation_time/`
- For case studies: `07_case_study_15N/`
- For general force analysis: Check the numbered poster sections

This folder is maintained for backward compatibility but will not be updated with new figures.
