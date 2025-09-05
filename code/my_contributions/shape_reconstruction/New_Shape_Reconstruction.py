"""Iterative shape reconstruction & depth calibration using SAM-2 segmentation.

This script unifies and modernises the original `_2_Sensor_Calibration.py` and
`_3_Shape_Reconstruction.py` pipelines.  The goal is to:
  1.  Auto–calibrate the pixel-intensity → depth mapping *without* the ball
      pressing procedure.  Instead, we use a short video where an object is
      pressed against the cam.  The shallowest (first) frame becomes the
      reference, the deepest frame determines the scale.
  2.  Rely on the contact-region detector (high-pass + threshold) **and** an
      optional SAM-2 automatic segmenter to isolate only the contacting object
      pixels so that out-of-contact background does not influence calibration.
  3.  After calibration, reconstruct the 3-D height map for a subset of frames
      and visualise the resulting point cloud in Open3D.

Typical usage
-------------
python New_Shape_Reconstruction.py \
    --video calibration/frames/Screw_1.mp4 \
    --cfg   shape_config.yaml       \
    --sensor_id 3                  \
    --start  0 --end  600          \
    --samples 15                   \
    --max_depth 3.0                \
    --use_sam                      

Note: the first frame in the selected range is considered *shallow / no depth*.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple
from pathlib import Path

import cv2
import numpy as np
import yaml

# -----------------------------------------------------------------------------
#  Helper: rectify + safe crop
# -----------------------------------------------------------------------------

def safe_rect_crop(cam: 'Camera', bgr: np.ndarray) -> np.ndarray:
    """Return rectified frame; crop to ROI. Fallback to full rectified image if
    the crop window is outside bounds (prevents empty arrays)."""
    rect = cam.rectify_image(bgr)
    crop = cam.crop_image(rect)
    if crop.size == 0:
        return rect
    # Optional extra center-crop to suppress background beyond the gel
    ratio = getattr(cam, "extra_center_crop_ratio", 1.0)
    try:
        r = float(ratio)
    except Exception:
        r = 1.0
    r = max(0.1, min(1.0, r))
    if r < 1.0:
        h, w = crop.shape[:2]
        ch, cw = int(round(h * r)), int(round(w * r))
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        crop = crop[y0:y0+ch, x0:x0+cw]
    return crop

# Local modules – support both package and script execution
try:
    from .New_Camera import Camera  # type: ignore
    from .visualizer import Visualizer  # type: ignore
except Exception:
    from New_Camera import Camera
    from visualizer import Visualizer

# Optional – SAM-2 is imported lazily inside segment_image_sam2
build_sam2 = None  # type: ignore
SAM2AutomaticMaskGenerator = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
#  Contact-region detector (copied from New_Contact_Region_Detection.py)
# ──────────────────────────────────────────────────────────────────────────────

def detect_contact_region(gray: np.ndarray,
                          hp_blur: int = 51,
                          threshold: int = 25,
                          morph_ks: int = 5) -> np.ndarray:
    """Return a binary mask of pixels in contact with an object.

    Parameters
    ----------
    gray : np.ndarray
        Cropped grayscale input (uint8).
    hp_blur : int
        Kernel size (odd) for Gaussian blur used as background estimate.
    threshold : int
        Minimum |difference| to background to mark a pixel as contact.
    morph_ks : int
        Kernel size for morphological open/close operations.
    """
    bg = cv2.GaussianBlur(gray, (hp_blur, hp_blur), 0)
    diff = gray.astype(np.int16) - bg.astype(np.int16)
    mag = np.abs(diff).astype(np.uint8)
    _, mask = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks, morph_ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask  # uint8, {0,255}


# ──────────────────────────────────────────────────────────────────────────────
#  SAM-2 helper – very similar to New_Contact_Region_Detection.segment_image_sam2
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_sam2_paths(preferred_root: str | None = None,
                        ckpt_path: str | None = None,
                        cfg_path: str | None = None) -> tuple[str, str] | None:
    import os
    # explicit override wins
    if ckpt_path and cfg_path and os.path.isfile(ckpt_path) and os.path.isfile(cfg_path):
        return cfg_path, ckpt_path
    # env overrides
    env_ckpt = os.getenv("SAM2_CKPT")
    env_cfg = os.getenv("SAM2_CFG")
    if env_ckpt and env_cfg and os.path.isfile(env_ckpt) and os.path.isfile(env_cfg):
        return env_cfg, env_ckpt
    # root search
    roots = []
    if preferred_root:
        roots.append(preferred_root)
    # repo-level sibling directory layout: <root>/sam2
    script_dir = os.path.dirname(os.path.abspath(__file__))
    roots.append(os.path.normpath(os.path.join(script_dir, "..", "sam2")))
    # env root
    env_root = os.getenv("SAM2_ROOT")
    if env_root:
        roots.append(env_root)
    ckpt_candidates = [
        "sam2.1_hiera_large.pt",
        "sam2_hiera_large.pt",
        "sam2.1_hiera_large.pth",
    ]
    cfg_candidates = [
        os.path.join("sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
        os.path.join("sam2", "sam2.1_hiera_l.yaml"),
        os.path.join("sam2", "sam2_hiera_l.yaml"),
        os.path.join("sam2", "sam2", "sam2_hiera_l.yaml"),
    ]
    for root in roots:
        for c in ckpt_candidates:
            ck = os.path.join(root, c)
            if os.path.isfile(ck):
                for cf in cfg_candidates:
                    cg = os.path.join(root, cf)
                    if os.path.isfile(cg):
                        return cg, ck
    return None


def segment_image_sam2(bgr: np.ndarray,
                       device: str = "cpu",
                       sam2_root: str | None = None,
                       sam2_ckpt: str | None = None,
                       sam2_cfg: str | None = None) -> list:
    """Run SAM-2 automatic segmentation and return annotations list.

    The first call initialises the model and caches the mask generator, so
    subsequent calls are cheap.
    """
    if not hasattr(segment_image_sam2, "_generator"):
        # Lazy initialisation
        try:
            global build_sam2, SAM2AutomaticMaskGenerator  # type: ignore
            if build_sam2 is None:
                from sam2.build_sam import build_sam as _build
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as _Gen
                build_sam2 = _build
                SAM2AutomaticMaskGenerator = _Gen
        except Exception:
            print("[WARN] SAM-2 not installed – segmentation disabled.")
            segment_image_sam2._generator = None  # type: ignore[attr-defined]
        else:
            resolved = _resolve_sam2_paths(sam2_root, sam2_ckpt, sam2_cfg)
            if not resolved:
                print("[WARN] SAM-2 checkpoint/config not found – set SAM2_ROOT or pass --sam2_root/--sam2_ckpt/--sam2_cfg.")
                segment_image_sam2._generator = None  # type: ignore[attr-defined]
            else:
                cfg_path, ckpt_path = resolved
                print(f"[INFO] Loading SAM-2 from:\n  cfg: {cfg_path}\n  ckpt: {ckpt_path}")
                sam2_model = build_sam2(cfg_path, ckpt_path, device=device, apply_postprocessing=False)
                segment_image_sam2._generator = SAM2AutomaticMaskGenerator(  # type: ignore[attr-defined]
                    sam2_model,
                    points_per_side=32,
                    points_per_batch=64,
                    output_mode="binary_mask",
                )

    generator = getattr(segment_image_sam2, "_generator", None)
    if generator is None:
        return []  # segmentation disabled / unavailable

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return generator.generate(rgb)


# ──────────────────────────────────────────────────────────────────────────────
#  Video-frame utilities
# ──────────────────────────────────────────────────────────────────────────────

def sample_video_frames(video_path: str, start: int, end: int | None, n_samples: int) -> List[Tuple[int, np.ndarray]]:
    """Return *(index, BGR image)* tuples uniformly sampled from *start*…*end*.
    The first returned frame is always *start* so that we know the shallow frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, start)
    end   = total_frames - 1 if end is None or end < 0 else min(end, total_frames - 1)
    if start >= end:
        raise ValueError("Invalid frame range – *start* must be < *end*.")

    # Always include the start frame, then spread the rest uniformly
    indices = np.linspace(start, end, n_samples, dtype=int)
    indices[0] = start  # guarantee
    frames: List[Tuple[int, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Failed to read frame {idx}. Skipping.")
            continue
        frames.append((idx, frame))
    cap.release()
    if not frames:
        raise RuntimeError("No frames could be read from the video.")
    return frames


# ──────────────────────────────────────────────────────────────────────────────
#  Depth-calibration from sampled frames
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_depth_from_frames(frames: List[Tuple[int, np.ndarray]],
                                cam: Camera,
                                contact_save_dir: str,
                                max_depth_mm: float = 3.0,
                                use_sam: bool = True,
                                sam2_kwargs: dict | None = None) -> Tuple[int, int]:
    """Analyse *frames* list and update *sensor* with a new Pixel→Depth mapping.

    The first frame is assumed to be the reference (no contact).  Only pixels
    inside the contact mask are considered when computing the *min*/*max*
    intensity differences.

    Returns
    -------
    diff_min, diff_max : int
        Extreme pixel-intensity differences that define the calibration range.
    """
    # ─── Reference (shallow) frame ────────────────────────────────────────────
    ref_idx, ref_bgr = frames[0]
    ref_undist = safe_rect_crop(cam, ref_bgr)
    ref_gray   = cv2.cvtColor(ref_undist, cv2.COLOR_BGR2GRAY)

    # Override the sensor reference so that downstream functions use *our* frame
    cam.ref       = ref_undist
    cam.ref_GRAY  = ref_gray

    collected_diffs: List[int] = []
    diff_min, diff_max = 255, 0

    for idx, bgr in frames[1:]:  # skip reference itself
        img_rc = safe_rect_crop(cam, bgr)
        gray   = cv2.cvtColor(img_rc, cv2.COLOR_BGR2GRAY)

        # ─── SAM-2 segmentation as the only region of interest ────────────
        segs = segment_image_sam2(img_rc, **(sam2_kwargs or {})) if use_sam else []
        if not segs:
            # No segments available → fall back to entire frame
            seg_union = np.ones(gray.shape, dtype=bool)
        else:
            seg_union = np.zeros(gray.shape, dtype=bool)
            for ann in segs:
                seg_m = ann["segmentation"]
                if seg_m.shape != seg_union.shape:
                    seg_m = cv2.resize(seg_m.astype(np.uint8), seg_union.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                seg_union = np.logical_or(seg_union, seg_m)

        # ─── Evaluate intensity change only inside the segments ────────────
        diff = ref_gray.astype(np.int16) - gray.astype(np.int16)
        diff[diff < 0] = 0  # ignore brighter pixels (no contact)
                # save contact visualisation
        if not os.path.isdir(contact_save_dir):
            os.makedirs(contact_save_dir, exist_ok=True)
        overlay = img_rc.copy()
        overlay[~seg_union] = 0
        cv2.imwrite(os.path.join(contact_save_dir, f"{idx:05d}_contact.png"), overlay)

        contact_diffs = diff[seg_union]
        contact_diffs = contact_diffs[contact_diffs > 0]
        if contact_diffs.size == 0:
            continue

        diff_min = min(diff_min, int(contact_diffs.min()))
        diff_max = max(diff_max, int(contact_diffs.max()))
        collected_diffs.extend(contact_diffs.tolist())

    if not collected_diffs:
        raise RuntimeError("Could not find any contact pixels across the sampled frames.")

    if diff_max == diff_min:
        raise RuntimeError("Contact pixels have zero intensity range – cannot calibrate.")

    print(f"[INFO] Depth calibration – diff range: {diff_min} → {diff_max}  (Δ={diff_max - diff_min})")

    # ─── Build Pixel→Depth LUT ───────────────────────────────────────────────
    lut_size = diff_max + 1  # inclusive
    pixel_to_depth = np.zeros(lut_size, dtype=np.float32)

    for g in range(diff_min, diff_max + 1):
        pixel_to_depth[g] = (g - diff_min) / (diff_max - diff_min) * max_depth_mm

    # Update sensor in-memory attributes so that existing methods work
    cam.lighting_threshold = diff_min
    cam.Pixel_to_Depth     = pixel_to_depth
    cam.max_index          = lut_size - 1

    # Persist the LUT so that future sessions can load it automatically
    os.makedirs(cam.depth_calibration_dir, exist_ok=True)
    np.save(os.path.join(cam.depth_calibration_dir, "Pixel_to_Depth_iterative.npy"), pixel_to_depth)
    print(f"[INFO] Saved new Pixel_to_Depth LUT → {cam.depth_calibration_dir}.")

    return diff_min, diff_max


# ──────────────────────────────────────────────────────────────────────────────
#  Height-map utilities (no Sensor dependency)
# ──────────────────────────────────────────────────────────────────────────────

def initialise_depth_utils(cam: Camera, cfg: dict) -> None:
    """Attach depth-reconstruction attributes & point-cloud grid to *cam*."""
    recon_cfg = cfg["sensor_reconstruction"]
    cam.kernel_list = recon_cfg["kernel_list"]
    cam.contact_gray_base = recon_cfg["contact_gray_base"]
    cam.depth_k = recon_cfg["depth_k"]

    # Geometry grid (identical to original Sensor implementation)
    cam.expand_x = int(28.0 / cam.pixel_per_mm) + 2
    cam.expand_y = int(21.0 / cam.pixel_per_mm) + 2
    X, Y = np.meshgrid(np.arange(cam.expand_x), np.arange(cam.expand_y))
    pts = np.zeros((cam.expand_x * cam.expand_y, 3))
    pts[:, 0] = X.flatten() * cam.pixel_per_mm
    pts[:, 1] = -Y.flatten() * cam.pixel_per_mm
    pts[:, 2] = 0
    cam.points = pts


def raw_image_2_height_map(cam: Camera, img_gray: np.ndarray) -> np.ndarray:
    diff_raw = cam.ref_GRAY - img_gray - cam.lighting_threshold
    diff_mask = (diff_raw < 100).astype(np.uint8)
    diff = diff_raw * diff_mask + cam.lighting_threshold
    diff[diff > cam.max_index] = cam.max_index
    diff = cv2.GaussianBlur(diff.astype(np.float32), (7, 7), 0).astype(int)
    height_map = cam.Pixel_to_Depth[diff] - cam.Pixel_to_Depth[cam.lighting_threshold]
    for k in cam.kernel_list:
        height_map = cv2.GaussianBlur(height_map.astype(np.float32), (k, k), 0)
    return height_map


def height_map_2_depth_map(cam: Camera, height_map: np.ndarray) -> np.ndarray:
    contact_show = np.zeros_like(height_map)
    contact_show[height_map > 0] = cam.contact_gray_base
    depth_map = height_map * cam.depth_k + contact_show
    return depth_map.astype(np.uint8)


def expand_image(img: np.ndarray, cam: Camera) -> np.ndarray:
    img_expand = np.zeros((cam.expand_y, cam.expand_x), dtype=img.dtype)
    # If the incoming image is larger than the canvas computed from calibration,
    # fall back to returning the image itself – this prevents broadcasting errors
    if img.shape[0] > cam.expand_y or img.shape[1] > cam.expand_x:
        return img.copy()

    start_y = (cam.expand_y - img.shape[0]) // 2
    start_x = (cam.expand_x - img.shape[1]) // 2
    img_expand[start_y:start_y + img.shape[0], start_x:start_x + img.shape[1]] = img
    return img_expand


def height_map_2_point_cloud_gradients(height_map: np.ndarray, cam: Camera):
    dzdy, dzdx = np.gradient(height_map)  # note: numpy returns (dy, dx)
    h, w = height_map.shape
    # Ensure point grid matches incoming height map
    expected = h * w
    if getattr(cam, "points", None) is None or cam.points.shape[0] != expected:
        cam.expand_x = w
        cam.expand_y = h
        X, Y = np.meshgrid(np.arange(cam.expand_x), np.arange(cam.expand_y))
        pts_new = np.zeros((expected, 3))
        pts_new[:, 0] = X.flatten() * cam.pixel_per_mm
        pts_new[:, 1] = -Y.flatten() * cam.pixel_per_mm
        pts_new[:, 2] = 0
        cam.points = pts_new
    pts = cam.points.copy()
    pts[:, 2] = height_map.flatten()
    return pts, (dzdx, dzdy)


# ──────────────────────────────────────────────────────────────────────────────
#  Main routine
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative depth calibration + 3-D reconstruction")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video recorded while pressing an object on the cam.")
    parser.add_argument("--cfg", type=str, default="shape_config.yaml",
                        help="Sensor YAML configuration file.")
    parser.add_argument("--sensor_id", type=int, default=1,
                        help="Sensor index → selects the calibration folder.")
    parser.add_argument("--start", type=int, default=0, help="Start frame.")
    parser.add_argument("--end",   type=int, default=-1, help="End frame (-1 → last).")
    parser.add_argument("--samples", type=int, default=20,
                        help="How many frames to sample between start & end (including the start frame).")
    parser.add_argument("--max_depth", type=float, default=4.0,
                        help="Maximum indentation depth (mm) corresponding to the largest intensity diff.")
    parser.add_argument("--center_crop_ratio", type=float, default=0.6,
                        help="Additional center crop applied after rect+crop (0<r<=1, e.g., 0.6 keeps center 60%).")
    parser.add_argument("--use_sam", action="store_true", help="Use SAM-2 to refine the contact mask.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device on which to run SAM-2.")
    parser.add_argument("--sam2_root", type=str, default=None, help="Root folder that contains SAM-2 assets")
    parser.add_argument("--sam2_ckpt", type=str, default=None, help="Explicit path to SAM-2 checkpoint")
    parser.add_argument("--sam2_cfg", type=str, default=None, help="Explicit path to SAM-2 config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ─── Load configuration & construct *un-calibrated* sensor ───────────────
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["sensor_id"] = args.sensor_id

    # Resolve calibration_root_dir relative to YAML location if needed
    cfg_path = Path(args.cfg).resolve()
    cal_root = Path(cfg["calibration_root_dir"]) if "calibration_root_dir" in cfg else None
    if cal_root is not None and not cal_root.is_absolute():
        cfg["calibration_root_dir"] = str((cfg_path.parent / cal_root).resolve())

    cam = Camera(cfg, calibrated=True, file_mode=True)  # we will calibrate depth ourselves
    # apply extra center crop ratio to suppress background
    cam.extra_center_crop_ratio = float(args.center_crop_ratio)

    # ─── Depth reconstruction helper initialisation ──────────────────────
    depth_calib_dir = f"{cfg['calibration_root_dir']}/sensor_{args.sensor_id}{cfg['depth_calibration']['depth_calibration_dir']}"
    cam.depth_calibration_dir = depth_calib_dir
    initialise_depth_utils(cam, cfg)  # attach kernel, grid, etc.

    # ─── Output directories ────────────────────────────────────────────
    out_3d_dir = os.path.join(cfg['calibration_root_dir'], '3Dmodel_2')
    contact_dir = os.path.join(cfg['calibration_root_dir'], 'ContactDetection')
    os.makedirs(out_3d_dir, exist_ok=True)
    os.makedirs(contact_dir, exist_ok=True)

    # ─── 1.  Sample frames & calibrate depth LUT ─────────────────────────────
    frames = sample_video_frames(args.video, args.start, args.end, args.samples)
    sam2_kwargs = dict(device=args.device, sam2_root=args.sam2_root, sam2_ckpt=args.sam2_ckpt, sam2_cfg=args.sam2_cfg)
    calibrate_depth_from_frames(frames, cam, contact_dir, max_depth_mm=args.max_depth, use_sam=args.use_sam, sam2_kwargs=sam2_kwargs)

    # ─── 2.  Visualiser (Open3D) set-up ──────────────────────────────────────
    # Render with clear contact frame
    visualizer = Visualizer(cam.points, show_sensor=True, use_transparent_contact=True)

    # prepare video writer for depth maps
    video_path = os.path.join(out_3d_dir, 'depth_reconstruction.avi')
    depth_writer = None

    # ─── 3.  Reconstruct & show sampled frames ──────────────────────────────
    for idx, bgr in frames:
        img_rc   = safe_rect_crop(cam, bgr)
        gray     = cv2.cvtColor(img_rc, cv2.COLOR_BGR2GRAY)
        height_m = raw_image_2_height_map(cam, gray)
        depth_m  = height_map_2_depth_map(cam, height_m)

        # save depth map & video
        cv2.imwrite(os.path.join(out_3d_dir, f"{idx:05d}_depth.png"), depth_m)
        if depth_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            depth_writer = cv2.VideoWriter(video_path, fourcc, 10, (depth_m.shape[1], depth_m.shape[0]), False)
        depth_writer.write(depth_m)

        cv2.imshow("Depth-map", depth_m)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        height_exp = expand_image(height_m, cam)
        pts, grads = height_map_2_point_cloud_gradients(height_exp, cam)

        if not visualizer.vis.poll_events():
            break
        visualizer.update(pts, grads)
        # save a PNG of the 3-D view
        pcd_fname = os.path.join(out_3d_dir, f"{idx:05d}_pcd.png")
        visualizer.vis.update_renderer()                # make sure latest frame is rendered
        visualizer.vis.capture_screen_image(pcd_fname)  # write PNG
    cv2.destroyAllWindows()
    if depth_writer is not None:
        depth_writer.release()
    visualizer.vis.destroy_window()


if __name__ == "__main__":
    main()
