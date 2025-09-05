"""Generate a poster-ready graph mapping force and deformation over time.

Theme: CUHK-style purple/gold with viridis accents to match existing figures.
Output: 9DTact/force_estimation/poster_figures/force_deformation_over_time.png
"""

from __future__ import annotations

import os
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Data (sequence order used as time index)
    start_depth_mm = 21.25
    pairs: List[tuple[float, float]] = [
        (0.55, 21.15),
        (1.092, 21.05),
        (1.509, 20.95),
        (2.002, 20.85),
        (3.07, 20.75),
        (3.503, 20.62),
        (4.492, 20.35),
        (5.51, 20.15),
        (6.06, 20.05),
        (7.02, 19.90),
        (8.08, 19.70),
        (9.05, 19.55),
        (10.00, 19.45),
        (10.95, 19.15),
        (12.0, 19.00),
        (12.95, 18.95),
        (13.96, 18.85),
        (15.02, 18.75),
        (17.03, 18.50),
        (18.75, 18.20),
        (20.02, 18.00),
    ]

    force_N = np.array([p[0] for p in pairs], dtype=float)
    depth_mm = np.array([p[1] for p in pairs], dtype=float)
    deformation_mm = start_depth_mm - depth_mm  # positive indentation (negative Z)
    t = np.arange(len(force_N))  # time index (sequence)

    # CUHK-esque palette
    cuhk_purple = "#582C83"
    cuhk_gold = "#FFD100"
    cuhk_purple_2 = "#A1045A"
    accent = mpl.cm.viridis(np.linspace(0.15, 0.85, len(t)))

    plt.rcParams.update({
        "axes.titlesize": 4,
        "axes.labelsize": 6,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "grid.alpha": 0.25,
    })

    # Smaller canvas to fit poster layout
    fig, ax_left = plt.subplots(figsize=(6.0, 2.6), constrained_layout=True)

    # Left y-axis: deformation (mm)
    ax_left.plot(t, deformation_mm, color=cuhk_purple, linewidth=2.5, label="Deformation (mm)")
    ax_left.scatter(t, deformation_mm, c=accent, s=35, zorder=3)
    ax_left.set_xlabel("Time (sequence index)")
    ax_left.set_ylabel("Deformation (mm)", color=cuhk_purple)
    ax_left.tick_params(axis="y", labelcolor=cuhk_purple)
    ax_left.grid(True, which="both", axis="both")

    # Right y-axis: force (N)
    ax_right = ax_left.twinx()
    ax_right.plot(t, force_N, color=cuhk_purple_2, linewidth=2.0, linestyle="--", label="Force (N)")
    ax_right.scatter(t, force_N, c=accent, s=20, edgecolor="none", alpha=0.9, zorder=3)
    ax_right.set_ylabel("Force (N)", color=cuhk_purple_2)
    ax_right.tick_params(axis="y", labelcolor=cuhk_purple_2)

    # Title and legend (omit big title to save space; caption will be in poster)
    # Compose a combined legend
    handles = [
        mpl.lines.Line2D([0], [0], color=cuhk_purple, lw=2.5, label="Deformation (mm)"),
        mpl.lines.Line2D([0], [0], color=cuhk_purple, lw=2.0, ls="--", label="Force (N)"),
    ]
    ax_left.legend(handles=handles, loc="upper left")

    # Save
    out_dir = os.path.join("9DTact", "force_estimation", "poster_figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "force_deformation_over_time.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"[OK] Saved figure → {out_path}")


if __name__ == "__main__":
    main()


