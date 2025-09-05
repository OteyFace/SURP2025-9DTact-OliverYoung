# Figures

## Directory Structure

```
figures/
├── poster_figures/
│ ├── 01_hardware_setup/
│ ├── 02_calibration_distortion/
│ ├── 03_contact_region_detection/
│ ├── 04_force_deformation_time/
│ ├── 05_depth_reconstruction/
│ ├── 06_force_depth_dataset/
│ ├── 07_case_study_15N/
│ └── force_estimation/ # deprecated
├── generated/
├── referenced/
└── README.md
```

## Poster Figures (by section)

**01_hardware_setup/**
- `hardware_overview.png`
- `README.md`

**02_calibration_distortion/**
- `contact_detection_results.png`
- `README.md`

**03_contact_region_detection/**
- `comprehensive_analysis.png`
- `README.md`

**04_force_deformation_time/**
- `force_deformation_over_time.png`
- `README.md`

**05_depth_reconstruction/**
- `00076_raw.png`, `00076_depth.png`, `00076_pcd.png`
- `README.md`

**06_force_depth_dataset/**
- `contact_detection_results.png`, `sam2_depth_map.png`
- `README.md`

**07_case_study_15N/**
- `15N_WIN_20250811_16_58_52_Pro_contact_results.png`
- `15N_WIN_20250811_16_58_52_Pro_poster.png`
- `README.md`

**Legacy (`poster_figures/force_estimation/`)**
- **Deprecated**: use the numbered sections above instead.
  - Temporal analysis → `04_force_deformation_time/`
  - 15N case study → `07_case_study_15N/`

## Generated (analysis results)
- `contact_accuracy_comparison_SAM2.png`
- `comprehensive_analysis.png`
- `opencv_vs_sam2_depth_maps.png`

## Referenced
Figures from cited papers with attribution (see each file’s note).

## Usage
1. Use figures from `poster_figures/01–07`.
2. Each figure includes a ~100-word, self-contained caption in the section README.
3. Keep high-res (≥300 DPI) PNG, consistent naming, and provenance notes.
4. Attribute all external figures in `referenced/`.
5. `force_estimation/` kept only for backward compatibility.

## Citation
- Project figures: “Author’s own work, SURP2025-9DTact”.
- Referenced figures: use the specific citation in `referenced/`.
- Original 9DTact: Lin et al., IEEE RA-L (2023).

## Technical Specs
- Format: PNG (lossless)
- Resolution: ≥300 DPI
- Color space: RGB
