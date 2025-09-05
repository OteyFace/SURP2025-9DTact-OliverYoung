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
See `docs/installation.md` for setup instructions.

## Usage
See `docs/methodology.md` for usage instructions.

## Future Development
- Gather more video/photo data to process and create depth maps from  
- Use datasets for training and regression models  
- Explore extensions toward tactile–visual fusion  

---

## Acknowledgements
I thank **Prof. Hongliang Ren** for continuous support and for providing all resources needed to make this work possible.  
I am especially grateful to **Zhang Tao**, who offered exceptional guidance throughout the program and ensured the equipment and environment were ready for experimentation. Without the mentorship of Prof. Ren and Zhang Tao, this project would not have been possible.  

---

## References
- Luu, Q.K., Nguyen, D.Q., Nguyen, N.H., Dam, N.P., Ho, V.A. *Vision-based Proximity and Tactile Sensing for Robot Arms: Design, Perception, and Control.* Project website: [https://quan-luu.github.io/protac-website](https://quan-luu.github.io/protac-website).  
- Lin, C., Zhang, H., Xu, J., Wu, L., Xu, H. *9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation.* Project site: [https://linchangyi1.github.io/9DTact](https://linchangyi1.github.io/9DTact). arXiv:2308.14277.  
- IEEE Xplore document (arnumber 11027485): [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11027485](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11027485).  

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact
[Oliver Young](mailto:ojyoung@ucsc.edu)
