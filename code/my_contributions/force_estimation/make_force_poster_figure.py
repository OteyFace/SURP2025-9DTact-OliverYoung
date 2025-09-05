"""
Generate poster-ready multi-panel PNGs showing the analysis pipeline for
ForceContactImages. For each input image we create a 2x3 grid similar to
contact_detection_results.png with:

Row 1: raw BGR, rectified+cropped, grayscale
Row 2: contact mask, depth map visualization, 3D force visualization screenshot

We rely on existing calibration (row/col maps and Pixel_to_Depth LUTs) and the
simple contact detector used by the dataset generation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import yaml

# Ensure project root (…/9DTact) on PYTHONPATH so `import shape_reconstruction.…` works
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shape_reconstruction.New_Camera import Camera
from shape_reconstruction.New_Shape_Reconstruction import (
    initialise_depth_utils,
    raw_image_2_height_map,
)
# Import helpers; support both module and script execution
try:
    from force_estimation.generate_force_depth_dataset import (
        simple_contact_mask,
        compute_depth_stats,
        parse_force_from_name,
    )
except Exception:  # fallback when executed as a module with relative imports broken
    from .generate_force_depth_dataset import (
        simple_contact_mask,
        compute_depth_stats,
        parse_force_from_name,
    )
from shape_reconstruction.New_Shape_Reconstruction import segment_image_sam2


def ensure_absolute_cal_root(cfg: dict, cfg_path: Path) -> dict:
    cal_root = Path(cfg["calibration_root_dir"]) if "calibration_root_dir" in cfg else None
    if cal_root is not None and not cal_root.is_absolute():
        cfg["calibration_root_dir"] = str((cfg_path.parent / cal_root).resolve())
    return cfg


def load_sensor(cfg_path: str, sensor_id: int) -> Camera:
    cfg_p = Path(cfg_path).resolve()
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["sensor_id"] = sensor_id
    cfg = ensure_absolute_cal_root(cfg, cfg_p)

    cam = Camera(cfg, calibrated=True, file_mode=True)
    initialise_depth_utils(cam, cfg)

    depth_dir = Path(cfg["calibration_root_dir"]) / f"sensor_{sensor_id}" / cfg["depth_calibration"]["depth_calibration_dir"].lstrip("/")
    p2d_iter = depth_dir / "Pixel_to_Depth_iterative.npy"
    if p2d_iter.is_file():
        cam.Pixel_to_Depth = np.load(str(p2d_iter))
    else:
        p2d_default = depth_dir / Path(cfg["depth_calibration"]["Pixel_to_Depth_path"]).name
        cam.Pixel_to_Depth = np.load(str(p2d_default))
    cam.max_index = cam.Pixel_to_Depth.shape[0] - 1
    cam.lighting_threshold = cfg.get("sensor_reconstruction", {}).get("lighting_threshold", 2)
    return cam


def center_crop(img: np.ndarray, ratio: float) -> np.ndarray:
    """Return a centre crop keeping `ratio` of width and height (0<ratio<=1)."""
    ratio = max(0.05, min(1.0, float(ratio)))
    h, w = img.shape[:2]
    ch, cw = int(round(h * ratio)), int(round(w * ratio))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0:y0+ch, x0:x0+cw]


def make_panel_png(
    cam: Camera,
    ref_path: str,
    img_path: str,
    out_png: str,
    crop_ratio: float,
    hp_blur: int,
    threshold: int,
    morph_ks: int,
    post_dilate: int,
    keep_largest: bool,
) -> None:
    # Reference first
    cam.file_path = ref_path
    ref_bgr = cam.get_rectify_crop_image()
    ref_bgr = center_crop(ref_bgr, crop_ratio)
    cam.ref = ref_bgr
    cam.ref_GRAY = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    # Target
    cam.file_path = img_path
    bgr = cam.get_rectify_crop_image()
    bgr = center_crop(bgr, crop_ratio)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Build initial mask using tunables and high-pass method
    bg = cv2.GaussianBlur(gray, (hp_blur, hp_blur), 0)
    diff = gray.astype(np.int16) - bg.astype(np.int16)
    mag = np.abs(diff).astype(np.uint8)
    _, mask = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks, morph_ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Refine with SAM-2: keep only segments overlapping the contact mask
    try:
        segs = segment_image_sam2(bgr)
    except Exception:
        segs = []
    if segs:
        sam_union = np.zeros_like(mask, dtype=bool)
        for ann in segs:
            seg_m = ann["segmentation"]
            if seg_m.shape != mask.shape:
                seg_m = cv2.resize(seg_m.astype(np.uint8), mask.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            # only consider segments that intersect the contact mask
            if np.any(seg_m & (mask > 0)):
                sam_union |= seg_m
        if np.any(sam_union):
            mask = (sam_union & (mask > 0)).astype(np.uint8) * 255

    # Optional post-processing
    if keep_largest:
        num_labels, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
        if num_labels > 1:
            counts = np.bincount(labels.flatten())
            best = int(np.argmax(counts[1:]) + 1)
            mask = (labels == best).astype(np.uint8) * 255
    if post_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (post_dilate, post_dilate))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Depth map
    height_map = raw_image_2_height_map(cam, gray)
    # Visual depth (uint8)
    depth_viz = (height_map - height_map.min())
    if depth_viz.max() > 0:
        depth_viz = (depth_viz / depth_viz.max() * 255).astype(np.uint8)
    else:
        depth_viz = np.zeros_like(gray)

    # Force map from brightness within mask, normalized, scaled to N from filename
    mean_d, _, _, _ = compute_depth_stats(cam, bgr, mask)
    try:
        force_N = parse_force_from_name(Path(img_path).name)
    except Exception:
        force_N = 1.0
    mask_bool = mask > 0
    if np.any(mask_bool):
        roi = gray.astype(np.float32)
        roi_norm = roi - roi[mask_bool].min()
        denom = float(roi[mask_bool].max() - roi[mask_bool].min()) or 1.0
        roi_norm = roi_norm / denom  # 0..1 within mask
        force_map = np.zeros_like(roi_norm, dtype=np.float32)
        force_map[mask_bool] = roi_norm[mask_bool] * float(force_N)
    else:
        force_map = np.zeros_like(gray, dtype=np.float32)

    # 3D force visualization via Matplotlib surface (robust, no OpenGL)
    tmp_png = Path(out_png).with_suffix(".tmp_vis.png")
    try:
        fig = plt.figure(figsize=(6, 4), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        yy, xx = np.mgrid[0:force_map.shape[0], 0:force_map.shape[1]]
        stride = max(1, min(force_map.shape) // 100)
        surf = ax.plot_surface(xx[::stride, ::stride], yy[::stride, ::stride],
                               force_map[::stride, ::stride], cmap=cm.inferno,
                               linewidth=0, antialiased=False)
        ax.view_init(elev=40, azim=-135)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(tmp_png), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception:
        # fallback: blank image
        plt.imsave(str(tmp_png), np.zeros_like(force_map), cmap='inferno')

    # Compose a 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=200)
    axes[0, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Raw (rect+crop)"); axes[0, 0].axis('off')
    axes[0, 1].imshow(gray, cmap='gray'); axes[0, 1].set_title("Grayscale"); axes[0, 1].axis('off')
    # overlay mask on BGR
    overlay = bgr.copy(); overlay[mask == 0] = 0
    axes[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes[0, 2].set_title("Contact Isolated"); axes[0, 2].axis('off')

    axes[1, 0].imshow(mask, cmap='gray'); axes[1, 0].set_title("Contact Mask"); axes[1, 0].axis('off')
    axes[1, 1].imshow(depth_viz, cmap='inferno'); axes[1, 1].set_title(f"Depth Map (mean={mean_d:.3f} mm)"); axes[1, 1].axis('off')
    # 3D screenshot
    vis_img = cv2.imread(str(tmp_png), cv2.IMREAD_COLOR)
    if vis_img is None:
        vis_img = np.zeros((460, 640, 3), dtype=np.uint8)
    axes[1, 2].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)); axes[1, 2].set_title("3D Force Visualization"); axes[1, 2].axis('off')

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    try:
        tmp_png.unlink(missing_ok=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Create poster-ready force analysis panels from ForceContactImages")
    parser.add_argument("--cfg", default="shape_reconstruction/shape_config.yaml")
    parser.add_argument("--sensor_id", type=int, default=3)
    parser.add_argument("--img_dir", default="shape_reconstruction/calibration/ForceContactImages")
    parser.add_argument("--out_dir", default="force_estimation/poster_figures")
    parser.add_argument("--ref_hint", default="1N_", help="Substring to pick the lowest-force reference image")
    parser.add_argument("--crop_ratio", type=float, default=0.6, help="Centre-crop ratio (0-1], e.g. 0.6 keeps middle 60% of H and W")
    # Mask tuning
    parser.add_argument("--hp_blur", type=int, default=51, help="Gaussian kernel for background estimate (odd)")
    parser.add_argument("--threshold", type=int, default=5, help="Threshold for high-pass magnitude")
    parser.add_argument("--morph_ks", type=int, default=5, help="Morphological kernel size")
    parser.add_argument("--post_dilate", type=int, default=5, help="Post close/dilate to fill small gaps (0 to disable)")
    parser.add_argument("--keep_largest", action='store_true', help="Keep only the largest connected component")
    args = parser.parse_args()

    cam = load_sensor(args.cfg, args.sensor_id)

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not imgs:
        raise FileNotFoundError("No images found in image directory")

    # choose reference image: try hint first else lexicographically first
    ref_path = None
    for p in imgs:
        if args.ref_hint in p.name:
            ref_path = p
            break
    if ref_path is None:
        ref_path = imgs[0]

    for p in imgs:
        out_png = Path(args.out_dir) / f"{p.stem}_poster.png"
        make_panel_png(
            cam,
            str(ref_path),
            str(p),
            str(out_png),
            args.crop_ratio,
            args.hp_blur,
            args.threshold,
            args.morph_ks,
            args.post_dilate,
            args.keep_largest,
        )
        print(f"Saved {out_png}")


if __name__ == "__main__":
    main()


