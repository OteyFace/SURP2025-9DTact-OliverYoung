# Methodology

## Overview
This project extends the 9DTact tactile sensing pipeline with (1) SAM-2–assisted contact region detection, (2) a quantitative contact accuracy analysis suite, and (3) refinements to camera/sensor calibration and 3-D shape reconstruction.

## Components & Layout
- **Baseline (upstream 9DTact)**: `code/original_9dtact/` (or upstream clone)
- **New work**: `code/my_contributions/`
- **Data**: `data/raw/`, `data/processed/`, `data/analysis_results/`
- **Figures**: `figures/poster_figures/01–07/` (poster sections), `figures/generated/` (analysis plots)

> See `docs/installation.md` for environment setup and `docs/results.md` for outcomes.

---

## Pipelines

### A) Contact Accuracy Analysis
**Goal.** Quantify segmentation quality and geometry consistency across methods (SAM-2 vs OpenCV).

**Inputs.**
- Reference/benchmark image(s) (e.g., manually verified or consensus masks)
- Candidate images/masks from each method

**Metrics.**
- **Area** (px²), **Equivalent Diameter** (from area), **Perimeter**, **Circularity** (= \(4\pi \cdot \text{Area} / \text{Perimeter}^2\)),
- **Centroid offset** (px), **IoU**/**Dice** against benchmark,
- Optional: radial profile, convexity/solidity.

**Script.**
- `code/my_contributions/contact_accuracy_analysis.py`

**Run.**
```bash
python code/my_contributions/contact_accuracy_analysis.py \
  --benchmark data/analysis_results/benchmark.png \
  --candidate_left data/analysis_results/candidate1.png \
  --candidate_right data/analysis_results/candidate2.png \
  --output_dir data/analysis_results/

Outputs.

CSV metrics → data/analysis_results/*.csv

Side-by-side plots / overlays → figures/generated/

Summary PNGs (comparisons) → figures/generated/

B) Shape Reconstruction Enhancements

Goal. Improve depth/shape fidelity by feeding cleaner contact masks and tightening calibration.

Improvements.

SAM-2 masks to delineate contact regions before depth inference

Refined camera calibration (intrinsics, distortion) and sensor calibration (gel thickness, ball radius, LUTs)

Post-processing: morphological cleanup, denoise, temporal smoothing

Scripts.

code/my_contributions/New_Shape_Reconstruction.py

Additional helpers matching New_*.py patterns

Run.

cd code/my_contributions/
python New_Shape_Reconstruction.py


Outputs.

Depth maps → data/processed/calibration/3DModel_*/**/*_depth.png

Point clouds / meshes → *_pcd.png / exports

Visuals for poster → figures/poster_figures/05_depth_reconstruction/

C) SAM-2 Integration

Purpose. Robust segmentation on gel images with low texture/contrast.

Model: Segment Anything 2 (SAM-2)

Usage: generate contact masks; optionally fine-tune thresholds/prompts

Interop: masks saved alongside frames for downstream reconstruction & metrics

Data Flow (End-to-End)

Raw capture → data/raw/Videos/

Frame extraction → data/processed/calibration/frames/<object>/

Camera & sensor calibration (intrinsics/distortion, gel/ball params)

Segmentation (SAM-2, OpenCV) → contact masks

Depth/shape reconstruction → depth maps, PCDs

Analysis → metrics (CSV) + generated plots

Figures → poster sections figures/poster_figures/01–07/

Reproducibility

Activate env: conda activate 9dtact

Fix seeds where applicable (NumPy, PyTorch)

Record script + commit in captions/CSVs (provenance)

If OpenCV GUI errors (Linux), install libgl1 ffmpeg

Notes

If your repo uses the original 9DTact layout (e.g., shape_reconstruction/_1_Camera_Calibration.py, _2_Sensor_Calibration.py, _3_Shape_Reconstruction.py), adjust paths accordingly in the commands above.

Keep heavy intermediate artifacts out of version control; store metrics/plots and small samples in-repo, and link large data externally.