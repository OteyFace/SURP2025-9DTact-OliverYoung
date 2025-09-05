import os
import re
import csv
from typing import List, Tuple

import cv2
import numpy as np
import yaml

from pathlib import Path

# -----------------------------------------------------------------------------
#  Ensure project root (…/9DTact) is on PYTHONPATH so that
#  `import shape_reconstruction.…` works even when the script is executed from
#  another directory (e.g. shape_reconstruction/).
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports – after PYTHONPATH fix
from shape_reconstruction.New_Camera import Camera
from shape_reconstruction.New_Shape_Reconstruction import initialise_depth_utils, raw_image_2_height_map


def parse_force_from_name(fname: str) -> float:
    """Extract the force value (in Newton) from a file name like ``'3.5N_...'``."""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)N", fname)
    if not m:
        raise ValueError(f"Could not parse force from filename: {fname}")
    return float(m.group(1))


def collect_image_paths(root_dir: str) -> List[str]:
    """Return a sorted list of *.jpg / *.png inside *root_dir*."""
    img_ext = {".jpg", ".jpeg", ".png"}
    paths = [str(p) for p in Path(root_dir).glob("*.*") if p.suffix.lower() in img_ext]
    return sorted(paths)


def simple_contact_mask(gray: np.ndarray, hp_blur: int = 51, threshold: int = 5, morph_ks: int = 5) -> np.ndarray:
    """Lightweight contact detector (no SAM).

    Returns a binary mask (uint8) where contact pixels are 255.
    """
    bg = cv2.GaussianBlur(gray, (hp_blur, hp_blur), 0)
    diff = gray.astype(np.int16) - bg.astype(np.int16)
    mag = np.abs(diff).astype(np.uint8)
    _, mask = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks, morph_ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def compute_depth_stats(cam: Camera, bgr: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, int]:
    """Return (mean_depth, max_depth, sum_depth, area_px) for *mask* pixels."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height_map = raw_image_2_height_map(cam, gray)  # mm per pixel

    # ensure mask matches height_map shape
    if mask.shape != height_map.shape:
        mask = cv2.resize(mask.astype(np.uint8), height_map.shape[::-1], interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask > 0)
    roi = height_map[mask_bin]
    if roi.size == 0:
        return 0.0, 0.0, 0.0, 0
    return float(roi.mean()), float(roi.max()), float(roi.sum()), int(mask_bin.sum())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate <force, depth_stats> CSV from labelled images")
    parser.add_argument("--cfg", default="shape_reconstruction/shape_config.yaml", help="Sensor YAML config (relative to project root or absolute)")
    parser.add_argument("--sensor_id", type=int, default=3, help="Sensor index matching calibration folder")
    parser.add_argument("--img_dir", default="shape_reconstruction/calibration/ForceContactImages", help="Directory with labelled force images (relative to project root or absolute)")
    parser.add_argument("--csv_out", default="force_estimation/force_depth_dataset.csv", help="Output CSV file path (relative to project root or absolute)")
    args = parser.parse_args()

    # ─── Load sensor (includes Pixel→Depth LUT) ───────────────────────────────
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["sensor_id"] = args.sensor_id

    # Make calibration_root_dir absolute relative to the YAML location if needed
    cal_root = Path(cfg["calibration_root_dir"]) if "calibration_root_dir" in cfg else None
    if cal_root is not None and not cal_root.is_absolute():
        cfg["calibration_root_dir"] = str((cfg_path.parent / cal_root).resolve())

    cam = Camera(cfg, calibrated=True, file_mode=True)  # we set file_path per-image later

    # depth utils (Pixel→Depth etc.)
    initialise_depth_utils(cam, cfg)

    # CSV header
    rows = [("filename", "force_N", "mean_depth_mm", "max_depth_mm", "sum_depth_mm", "area_px")]

    # iterate over images
    img_dir = Path(args.img_dir)
    if not img_dir.is_absolute():
        img_dir = (PROJECT_ROOT / img_dir).resolve()
    img_paths = collect_image_paths(str(img_dir))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {args.img_dir}")

    # Use the smallest-force image as the reference frame (assumed shallow)
    parsed = sorted([(parse_force_from_name(os.path.basename(p)), p) for p in img_paths], key=lambda x: x[0])
    ref_force, ref_path = parsed[0]
    cam.file_path = ref_path
    ref_bgr = cam.get_rectify_crop_image()
    cam.ref = ref_bgr
    cam.ref_GRAY = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    # depth calibration directory + LUT
    depth_dir = Path(cfg["calibration_root_dir"]) / f"sensor_{args.sensor_id}" / cfg["depth_calibration"]["depth_calibration_dir"].lstrip("/")
    cam.depth_calibration_dir = str(depth_dir)
    # Prefer iterative LUT if present, else default Pixel_to_Depth.npy
    p2d_iter = depth_dir / "Pixel_to_Depth_iterative.npy"
    if p2d_iter.is_file():
        cam.Pixel_to_Depth = np.load(str(p2d_iter))
    else:
        p2d_default = depth_dir / Path(cfg["depth_calibration"]["Pixel_to_Depth_path"]).name
        if not p2d_default.is_file():
            raise FileNotFoundError(f"Missing Pixel_to_Depth LUT in {depth_dir}. Run depth calibration first.")
        cam.Pixel_to_Depth = np.load(str(p2d_default))
    cam.max_index = cam.Pixel_to_Depth.shape[0] - 1

    # Lighting threshold from config (fallback to 2 if missing)
    cam.lighting_threshold = cfg.get("sensor_reconstruction", {}).get("lighting_threshold", 2)

    for p in img_paths:
        force_val = parse_force_from_name(os.path.basename(p))
        cam.file_path = p  # point camera to current file
        bgr = cam.get_rectify_crop_image()

        # contact mask (fast, no SAM)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = simple_contact_mask(gray, threshold=5)

        mean_d, max_d, sum_d, area_px = compute_depth_stats(cam, bgr, mask)
        rows.append((os.path.basename(p), force_val, mean_d, max_d, sum_d, area_px))
        print(f"Processed {os.path.basename(p):>30s} -> F={force_val:5.2f} N, mean_depth={mean_d:.3f} mm, area={area_px}")

    # save CSV
    csv_out = Path(args.csv_out)
    if not csv_out.is_absolute():
        csv_out = (PROJECT_ROOT / csv_out).resolve()
    os.makedirs(str(csv_out.parent), exist_ok=True)
    with open(str(csv_out), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved dataset → {csv_out}  ({len(rows)-1} samples)")


if __name__ == "__main__":
    main()
