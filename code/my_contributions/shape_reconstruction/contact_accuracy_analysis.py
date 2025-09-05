"""Contact width and area accuracy analysis.

This script compares the apparent contact size in different images against a
benchmark mask. It computes two metrics for each image: the diameter of the
largest contact region (via a minimum enclosing circle) and the contact area in
pixels. Results are summarized in a small report and saved as bar charts.

Usage example
-------------
python contact_accuracy_analysis.py \
  --benchmark shape_reconstruction/high_sensitivity_results/sam2_isolated_contact.png \
  --candidate_left shape_reconstruction/calibration/comprehensive_analysis/15N_WIN_20250811_16_58_52_Pro_contact_results.png \
  --candidate_right force_estimation/poster_figures/15N_WIN_20250811_16_58_52_Pro_poster.png \
  --output_dir shape_reconstruction/high_sensitivity_results \
  --label_left "15N contact (composite left)" \
  --label_right "15N 3D force visualization"

Notes
-----
- If you know the scale, pass --mm_per_px to convert diameters/areas to mm/mm^2.
- The segmentation is intentionally simple and robust: grayscale → Otsu
  threshold → largest contour → min enclosing circle. For composite figures,
  you may want to pre-crop to the panel of interest; optional crop arguments are
  provided per image.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shape_reconstruction.New_Contact_Region_Detection import segment_image_sam2


@dataclass
class CropRect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class ContactMetrics:
    diameter_px: float  # chosen diameter (see fields below)
    area_px: float
    center: Tuple[float, float]
    mask: np.ndarray
    perimeter_px: float
    circularity: float
    eq_diameter_px: float
    enclosing_diameter_px: float
    ellipse_major_px: float


def parse_crop(arg: Optional[str]) -> Optional[CropRect]:
    if not arg:
        return None
    try:
        parts = [int(p) for p in arg.split(",")]
        if len(parts) != 4:
            raise ValueError
        return CropRect(*parts)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            "Crop must be 'x,y,w,h' with integer values"
        ) from exc


def load_image_bgr(path: str) -> np.ndarray:
    """Robust image loader that resolves paths relative to this script if needed."""
    # Try as-is
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    script_dir = Path(__file__).parent.resolve()
    candidates = [
        script_dir / path,                         # relative to 9DTact/
        script_dir.parent / path,                  # relative to repo root
        Path(path.replace('9DTact/', '')),         # drop leading 9DTact/ if present
        script_dir / path.replace('9DTact/', ''),
    ]
    for cand in candidates:
        cand_str = str(cand)
        img = cv2.imread(cand_str, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    raise FileNotFoundError(f"Failed to read image: {path}")


def apply_optional_crop(image: np.ndarray, crop: Optional[CropRect]) -> np.ndarray:
    if crop is None:
        return image
    x0, y0, w, h = crop.x, crop.y, crop.w, crop.h
    x1, y1 = x0 + w, y0 + h
    x0 = max(0, min(x0, image.shape[1] - 1))
    y0 = max(0, min(y0, image.shape[0] - 1))
    x1 = max(1, min(x1, image.shape[1]))
    y1 = max(1, min(y1, image.shape[0]))
    return image[y0:y1, x0:x1]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood_padded, None, (0, 0), 255)
    flood = flood_padded[1 : h + 1, 1 : w + 1]
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, holes)
    return filled


def estimate_contact_mask(image_bgr: np.ndarray, debug_dir: Optional[str] = None, name: str = "") -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    unique_vals = np.unique(gray)
    if unique_vals.size <= 4:
        mask = np.where(gray > 0, 255, 0).astype(np.uint8)
    else:
        bg = cv2.GaussianBlur(gray, (31, 31), 0)
        hp = cv2.subtract(gray, bg)
        hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)
        _, m1 = cv2.threshold(hp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m2 = cv2.bitwise_not(m1)
        def largest_area(m: np.ndarray) -> float:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return 0.0 if not cnts else max(cv2.contourArea(c) for c in cnts)
        mask = m1 if largest_area(m1) >= largest_area(m2) else m2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _fill_holes(mask)

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_mask.png"), mask)
    return mask


def compute_contact_metrics(image_bgr: np.ndarray, debug_dir: Optional[str] = None, name: str = "") -> Optional[ContactMetrics]:
    mask = estimate_contact_mask(image_bgr, debug_dir, name)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(largest))
    perimeter_px = float(cv2.arcLength(largest, True))
    circularity = 0.0 if perimeter_px == 0 else 4.0 * np.pi * area_px / (perimeter_px**2)

    eq_diameter_px = 2.0 * np.sqrt(area_px / np.pi) if area_px > 0 else 0.0
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    enclosing_diameter_px = float(2.0 * radius)

    try:
        ellipse = cv2.fitEllipse(largest)
        (ex, ey), (maj, min_), angle = ellipse  # noqa: F841
        ellipse_major_px = float(max(maj, min_))
        center = (ex, ey)
    except cv2.error:
        ellipse_major_px = enclosing_diameter_px
        center = (cx, cy)

    if circularity >= 0.6:
        chosen = eq_diameter_px
    else:
        chosen = min(ellipse_major_px, enclosing_diameter_px)

    if debug_dir is not None:
        overlay = image_bgr.copy()
        cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 2)
        cv2.circle(overlay, (int(cx), int(cy)), int(enclosing_diameter_px / 2), (255, 0, 0), 2)
        cv2.circle(overlay, (int(center[0]), int(center[1])), int(chosen / 2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"{name}_overlay.png"), overlay)

    return ContactMetrics(
        diameter_px=chosen,
        area_px=area_px,
        center=(float(center[0]), float(center[1])),
        mask=mask,
        perimeter_px=perimeter_px,
        circularity=float(circularity),
        eq_diameter_px=float(eq_diameter_px),
        enclosing_diameter_px=float(enclosing_diameter_px),
        ellipse_major_px=float(ellipse_major_px),
    )


def solidify_center_region(mask: np.ndarray) -> np.ndarray:
    """Fill holes and convexify the largest central blob to form a solid contact area."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    largest = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(largest)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, [hull], -1, 255, thickness=cv2.FILLED)
    return solid


def metrics_from_mask(mask: np.ndarray, image_shape: Tuple[int, int]) -> ContactMetrics:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = image_shape
        return ContactMetrics(0.0, 0.0, (w / 2.0, h / 2.0), mask, 0.0, 0.0, 0.0, 0.0, 0.0)
    largest = max(cnts, key=cv2.contourArea)
    area_px = float(cv2.contourArea(largest))
    perimeter_px = float(cv2.arcLength(largest, True))
    circularity = 0.0 if perimeter_px == 0 else 4.0 * np.pi * area_px / (perimeter_px**2)
    eq_diameter_px = 2.0 * np.sqrt(area_px / np.pi) if area_px > 0 else 0.0
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    enclosing_diameter_px = float(2.0 * radius)
    try:
        ellipse = cv2.fitEllipse(largest)
        (ex, ey), (maj, min_), _ = ellipse
        ellipse_major_px = float(max(maj, min_))
        center = (ex, ey)
    except cv2.error:
        ellipse_major_px = enclosing_diameter_px
        center = (cx, cy)
    chosen = eq_diameter_px if circularity >= 0.6 else min(ellipse_major_px, enclosing_diameter_px)
    return ContactMetrics(
        diameter_px=float(chosen),
        area_px=area_px,
        center=(float(center[0]), float(center[1])),
        mask=mask,
        perimeter_px=perimeter_px,
        circularity=float(circularity),
        eq_diameter_px=float(eq_diameter_px),
        enclosing_diameter_px=float(enclosing_diameter_px),
        ellipse_major_px=float(ellipse_major_px),
    )


def auto_center_crop(image_bgr: np.ndarray, margin_ratio: float = 0.15) -> np.ndarray:
    tmp_mask = estimate_contact_mask(image_bgr)
    cnts, _ = cv2.findContours(tmp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image_bgr
    largest = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    mx = int(margin_ratio * w)
    my = int(margin_ratio * h)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(image_bgr.shape[1], x + w + mx)
    y1 = min(image_bgr.shape[0], y + h + my)
    return image_bgr[y0:y1, x0:x1]


def plot_comparison(
    benchmark: ContactMetrics,
    cand_left: Optional[ContactMetrics],
    cand_right: Optional[ContactMetrics],
    labels: Tuple[str, str, str],
    mm_per_px: Optional[float],
    output_dir: str,
) -> str:
    names = [labels[0], labels[1], labels[2]]
    diameters = [benchmark.diameter_px, np.nan, np.nan]
    areas = [benchmark.area_px, np.nan, np.nan]

    if cand_left is not None:
        diameters[1] = cand_left.diameter_px
        areas[1] = cand_left.area_px
    if cand_right is not None:
        diameters[2] = cand_right.diameter_px
        areas[2] = cand_right.area_px

    if mm_per_px is not None and mm_per_px > 0:
        diameters_plot = [d * mm_per_px if not np.isnan(d) else np.nan for d in diameters]
        areas_plot = [a * (mm_per_px**2) if not np.isnan(a) else np.nan for a in areas]
        diameter_unit = "mm"
        area_unit = "mm$^2$"
    else:
        diameters_plot = diameters
        areas_plot = areas
        diameter_unit = "px"
        area_unit = "px$^2$"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(names))
    axes[0].bar(x, diameters_plot, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_title(f"Contact diameter ({diameter_unit})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")

    bench_d = diameters_plot[0]
    for i in range(1, len(diameters_plot)):
        d = diameters_plot[i]
        if not np.isnan(d) and bench_d > 0:
            err = 100.0 * abs(d - bench_d) / bench_d
            axes[0].text(i, d, f"{err:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, areas_plot, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_title(f"Contact area ({area_unit})")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")

    bench_a = areas_plot[0]
    for i in range(1, len(areas_plot)):
        a = areas_plot[i]
        if not np.isnan(a) and bench_a > 0:
            err = 100.0 * abs(a - bench_a) / bench_a
            axes[1].text(i, a, f"{err:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "contact_accuracy_comparison_SAM2.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze_single(
    path: str,
    crop: Optional[CropRect],
    label: str,
    auto_crop: bool = False,
    crop_margin: float = 0.15,
) -> Optional[ContactMetrics]:
    image = load_image_bgr(path)
    image = apply_optional_crop(image, crop)
    if auto_crop:
        image = auto_center_crop(image, margin_ratio=crop_margin)
    debug_dir = None
    metrics = compute_contact_metrics(image, debug_dir, label.replace(" ", "_"))
    if metrics is None:
        print(f"[WARN] No contact region detected in '{label}' → {path}")
    else:
        print(
            (
                f"[INFO] {label}: chosen_diam={metrics.diameter_px:.2f}px | "
                f"eq_diam={metrics.eq_diameter_px:.2f}px | encl_diam={metrics.enclosing_diameter_px:.2f}px | "
                f"ellipse_major={metrics.ellipse_major_px:.2f}px | circ={metrics.circularity:.3f} | "
                f"area={metrics.area_px:.0f}px^2"
            )
        )
    return metrics


def unc_bolt_diameter_mm(gauge: int) -> float:
    inches_map = {2: 0.0860, 4: 0.1120, 6: 0.1380, 8: 0.1640, 10: 0.1900}
    if gauge not in inches_map:
        raise ValueError("Unsupported bolt gauge; supported: #2, #4, #6, #8, #10")
    return inches_map[gauge] * 25.4


def _scale_metrics(m: Optional[ContactMetrics], factor: float) -> Optional[ContactMetrics]:
    if m is None or factor == 1.0 or factor <= 0:
        return m
    return ContactMetrics(
        diameter_px=m.diameter_px * factor,
        area_px=m.area_px * (factor ** 2),
        center=m.center,
        mask=m.mask,
        perimeter_px=m.perimeter_px if hasattr(m, "perimeter_px") else 0.0,
        circularity=m.circularity if hasattr(m, "circularity") else 0.0,
        eq_diameter_px=m.eq_diameter_px * factor if hasattr(m, "eq_diameter_px") else 0.0,
        enclosing_diameter_px=m.enclosing_diameter_px * factor if hasattr(m, "enclosing_diameter_px") else 0.0,
        ellipse_major_px=m.ellipse_major_px * factor if hasattr(m, "ellipse_major_px") else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare contact size accuracy against a benchmark mask")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=os.path.join(
            "shape_reconstruction",
            "high_sensitivity_results",
            "opencv_isolated_contact.png",
        ),
        help="Path to the benchmark mask/image (default: OpenCV isolated contact)",
    )
    parser.add_argument(
        "--candidate_left",
        type=str,
        default=os.path.join(
            "shape_reconstruction",
            "calibration",
            "comprehensive_analysis",
            "15N_WIN_20250811_16_58_52_Pro_contact_results.png",
        ),
        help="Path to the left 15N contact composite",
    )
    parser.add_argument(
        "--candidate_right",
        type=str,
        default=os.path.join(
            "force_estimation",
            "poster_figures",
            "15N_WIN_20250811_16_58_52_Pro_poster.png",
        ),
        help="Path to the right 15N 3D force visualization image",
    )
    parser.add_argument(
        "--mm_per_px",
        type=float,
        default=None,
        help="Optional scale in millimeters per pixel for physical units",
    )
    parser.add_argument(
        "--bolt_gauge",
        type=int,
        default=None,
        help="If provided (e.g., 8 for #8), infer mm/px from benchmark diameter",
    )
    parser.add_argument(
        "--scale_benchmark_diam_px",
        type=float,
        default=None,
        help="If set, scale benchmark metrics so that its diameter equals this many pixels",
    )
    parser.add_argument(
        "--auto_center_crop",
        action="store_true",
        help="Auto-crop around the largest blob before measuring (helps composites)",
    )
    parser.add_argument(
        "--crop_margin",
        type=float,
        default=0.15,
        help="Relative margin added around auto-crop bounding box",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("shape_reconstruction", "high_sensitivity_results"),
        help="Directory to write the comparison figure",
    )

    parser.add_argument("--crop_bench", type=parse_crop, default=None, help="Optional crop for benchmark 'x,y,w,h'")
    parser.add_argument("--crop_left", type=parse_crop, default=None, help="Optional crop for left candidate 'x,y,w,h'")
    parser.add_argument("--crop_right", type=parse_crop, default=None, help="Optional crop for right candidate 'x,y,w,h'")

    parser.add_argument(
        "--labels",
        type=str,
        nargs=3,
        default=("Benchmark (OpenCV)", "15N contact (left)", "15N 3D force (right)"),
        help="Three labels for legend and axis ticks",
    )
    parser.add_argument(
        "--solidify_center",
        action="store_true",
        help="Make benchmark a solid central area (convex hull of largest blob)",
    )
    parser.add_argument(
        "--sam2_benchmark_from",
        type=str,
        default=None,
        help="Optional RGB path: build benchmark from largest SAM2 segment",
    )

    args = parser.parse_args()

    # Build benchmark metrics (optionally from SAM2 or solidified center)
    if args.sam2_benchmark_from is not None:
        bgr_bench = load_image_bgr(args.sam2_benchmark_from)
        segs = segment_image_sam2(bgr_bench)
        if segs:
            h, w = bgr_bench.shape[:2]
            best = max(segs, key=lambda a: np.count_nonzero(a["segmentation"]))
            seg_mask = best["segmentation"].astype(np.uint8)
            if seg_mask.shape != (h, w):
                seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            seg_mask = (seg_mask > 0).astype(np.uint8) * 255
            bench_metrics = metrics_from_mask(seg_mask, (h, w))
            print(f"[INFO] Benchmark from SAM2: area={bench_metrics.area_px:.0f} px^2, diam={bench_metrics.diameter_px:.2f} px")
        else:
            print("[WARN] SAM2 returned no segments; falling back to benchmark image masking.")
            bench_metrics = analyze_single(
                args.benchmark, args.crop_bench, args.labels[0], args.auto_center_crop, args.crop_margin
            )
    else:
        bench_metrics = analyze_single(
            args.benchmark, args.crop_bench, args.labels[0], args.auto_center_crop, args.crop_margin
        )

    if bench_metrics and args.solidify_center:
        solid = solidify_center_region(bench_metrics.mask)
        bench_metrics = metrics_from_mask(solid, solid.shape)
        print(f"[INFO] Solidified benchmark: area={bench_metrics.area_px:.0f} px^2, diam={bench_metrics.diameter_px:.2f} px")
    if bench_metrics is None:
        raise RuntimeError("Benchmark segmentation failed; provide a clearer image or a crop.")

    # Optional: rescale benchmark to target diameter in pixels
    if args.scale_benchmark_diam_px is not None and bench_metrics.diameter_px > 0:
        factor = float(args.scale_benchmark_diam_px) / float(bench_metrics.diameter_px)
        bench_metrics = _scale_metrics(bench_metrics, factor)  # type: ignore
        print(
            f"[INFO] Scaled benchmark to {args.scale_benchmark_diam_px:.1f}px (factor={factor:.4f})"
        )

    left_metrics = analyze_single(
        args.candidate_left, args.crop_left, args.labels[1], args.auto_center_crop, args.crop_margin
    )
    right_metrics = analyze_single(
        args.candidate_right, args.crop_right, args.labels[2], args.auto_center_crop, args.crop_margin
    )

    # infer scale if not provided
    mm_per_px = args.mm_per_px
    if mm_per_px is None and args.bolt_gauge is not None:
        known_diam_mm = unc_bolt_diameter_mm(args.bolt_gauge)
        mm_per_px = known_diam_mm / bench_metrics.diameter_px
        print(
            f"[INFO] Inferred scale from #{args.bolt_gauge} bolt: {known_diam_mm:.3f} mm / {bench_metrics.diameter_px:.2f} px → mm/px={mm_per_px:.6f}"
        )

    out_path = plot_comparison(
        benchmark=bench_metrics,
        cand_left=left_metrics,
        cand_right=right_metrics,
        labels=(args.labels[0], args.labels[1], args.labels[2]),
        mm_per_px=mm_per_px,
        output_dir=args.output_dir,
    )

    print(f"[OK] Saved comparison figure → {out_path}")


if __name__ == "__main__":
    main()


