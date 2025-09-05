# SURP2025-9DTact: Enhanced Contact Analysis and Shape Reconstruction

## Overview
This project extends the 9DTact tactile sensor system with enhanced contact analysis, SAM2 integration, and improved shape reconstruction capabilities.

## Original Work Attribution
This project builds upon the 9DTact sensor system:
- **Original Paper**: Lin, C., Zhang, H., Xu, J., Wu, L., & Xu, H. (2023). 9DTact: A compact vision-based tactile sensor for accurate 3D shape reconstruction and generalizable 6D force estimation. IEEE Robotics and Automation Letters, 9(2), 923-930.
- **Original Repository**: https://github.com/linchangyi1/9DTact
- **Original Authors**: Changyi Lin, Han Zhang, Jikai Xu, Lei Wu, Huazhe Xu

## My Contributions
- Enhanced contact region detection using SAM2
- Contact accuracy analysis framework
- Improved shape reconstruction algorithms
- Comprehensive analysis of sensor performance
- Integration of modern computer vision techniques

## Repository Structure
```
SURP2025-9DTact/
├── code/
│   ├── original_9dtact/          # Original 9DTact code (attributed)
│   └── my_contributions/         # My original work
├── data/                         # Research data - includes all pre- and post-processed images/videos used to generate poster figures
├── figures/                      # All figures organized by poster sections
│   └── poster_figures/           # Figures organized by poster sections
│       ├── 01_hardware_setup/    # Hardware & Setup
│       ├── 02_calibration_distortion/ # Calibration and distortion removal
│       ├── 03_contact_region_detection/ # Contact Region Detection
│       ├── 04_force_deformation_time/ # Force-Deformation over Time
│       ├── 05_depth_reconstruction/ # Depth Reconstruction
│       ├── 06_force_depth_dataset/ # Force-Depth Dataset and Regression
│       └── 07_case_study_15N/    # Case Study: 15N Contact
├── documentation/                # Complete documentation
├── poster/                       # Poster materials (LaTeX source and compiled PDF)
└── models/                       # Trained models
```

## Installation
See `documentation/installation.md` for setup instructions.

## Usage
See `documentation/methodology.md` for usage instructions.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
[Oliver Young] - [ojyoung@ucsc.edu]
