# Data

## Directory Structure

```
data/
├── raw/ # Original recordings
│ └── Videos/ # e.g., BluePill_1.mp4, Screw_1.mp4
├── processed/ # Calibrated / derived artifacts
│ ├── calibration/
│ │ ├── 3DModel/
│ │ ├── 3DModel_2/
│ │ ├── comprehensive_analysis/
│ │ ├── ContactDetection/
│ │ ├── ForceContactImages/
│ │ ├── frames/
│ │ ├── images/
│ │ ├── sensor_1/
│ │ ├── sensor_2/
│ │ ├── sensor_3/
│ │ └── WIN_20250703_15_55_38_Pro.mp4
└── analysis_results/
├── analysis_summary.csv
├── contact_accuracy_summary.csv
├── force_depth_dataset.csv
├── comprehensive_analysis.png
├── contact_accuracy_comparison_SAM2.png
├── opencv_depth_map.png
├── sam2_depth_map.png
├── opencv_isolated_contact.png
├── sam2_isolated_contact.png
└── README.md
```

## Contents

### `raw/`
- Original 9DTact videos (MP4, typically 30–120s).

### `processed/calibration/`
- `sensor_{1,2,3}/`: camera & depth calibration artifacts.
- `3DModel/`, `3DModel_2/`: depth maps, point clouds, video exports.
- `ContactDetection/`: masks, boundaries, morphology outputs.
- `ForceContactImages/`: images labeled by applied force.
- `frames/`: extracted frames grouped by object (e.g., `Screw_1/`).
- `images/`: rectified/processed calibration images.
- `comprehensive_analysis/`: cross-method comparisons.

### `analysis_results/`
- CSVs with metrics/datasets and figures comparing methods (OpenCV vs SAM2, depth maps, contact isolation).

## Processing Pipeline
1. Capture raw videos → `raw/Videos/`
2. Extract frames → `processed/calibration/frames/`
3. Camera rectification & distortion correction
4. Contact detection (OpenCV, SAM2) → masks/overlays
5. Depth/shape reconstruction → depth maps, point clouds
6. Metrics & summaries → `analysis_results/`

## Usage
- Poster figures reference datasets in `processed/` and metrics in `analysis_results/`.
- Calibration examples live under `processed/calibration/sensor_*`.
- Raw reproductions start from `raw/Videos/`.

## File Naming
- Videos: `{ObjectType}_{Number}.mp4`
- Frames: `{FrameNumber:05d}.jpg`
- Depth: `{FrameNumber:05d}_depth.png`
- Point clouds: `{FrameNumber:05d}_pcd.png`
- Analysis figs: `{analysis_type}_{method}.png`
- Tables: `*.csv`

## Specs
- Video: MP4 (H.264), RGB, 30–60 FPS
- Images: PNG (lossless), ≥1920×1080
- Tables: CSV

## Citation
- Project data/figures: “Author’s own work, SURP2025-9DTact”.
- 9DTact sources: Lin et al., IEEE RA-L (2023).
- SAM-2 usage: attribute Meta AI (see method docs).

## Integrity & Provenance
- Processed artifacts trace to raw files (paths in captions/CSVs).
- Sensor-specific calibration kept separated (`sensor_*`).
- Analysis includes reproducible inputs/outputs with versioned scripts.
