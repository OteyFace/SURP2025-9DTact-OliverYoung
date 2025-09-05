import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import yaml

from shape_reconstruction.New_Camera import Camera
from shape_reconstruction.New_Shape_Reconstruction import initialise_depth_utils, raw_image_2_height_map
from .generate_force_depth_dataset import simple_contact_mask, compute_depth_stats


class RuntimeForcePredictor:
    def __init__(self, cfg_path: str, sensor_id: int, coef_json: str):
        cfg_p = Path(cfg_path).resolve()
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg["sensor_id"] = sensor_id

        # Make calibration_root_dir absolute relative to YAML location if needed
        cal_root = Path(cfg["calibration_root_dir"])
        if not cal_root.is_absolute():
            cfg["calibration_root_dir"] = str((cfg_p.parent / cal_root).resolve())

        self.cam = Camera(cfg, calibrated=True, file_mode=True)
        initialise_depth_utils(self.cam, cfg)

        # Load coefficients
        with open(coef_json, "r", encoding="utf-8") as f:
            params = json.load(f)
        self.intercept = float(params["intercept"])    # in Newtons
        self.coef_mean = float(params["coef_mean_depth"])  # N per mm

        # Prepare depth LUT similarly to dataset pipeline
        depth_dir = Path(cfg["calibration_root_dir"]) / f"sensor_{sensor_id}" / cfg["depth_calibration"]["depth_calibration_dir"].lstrip("/")
        p2d_iter = depth_dir / "Pixel_to_Depth_iterative.npy"
        if p2d_iter.is_file():
            self.cam.Pixel_to_Depth = np.load(str(p2d_iter))
        else:
            p2d_default = depth_dir / Path(cfg["depth_calibration"]["Pixel_to_Depth_path"]).name
            self.cam.Pixel_to_Depth = np.load(str(p2d_default))
        self.cam.max_index = self.cam.Pixel_to_Depth.shape[0] - 1
        self.cam.lighting_threshold = cfg.get("sensor_reconstruction", {}).get("lighting_threshold", 2)

        self._is_ref_set = False

    def _ensure_reference(self, ref_image_path: str):
        if self._is_ref_set:
            return
        self.cam.file_path = ref_image_path
        bgr = self.cam.get_rectify_crop_image()
        self.cam.ref = bgr
        self.cam.ref_GRAY = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self._is_ref_set = True

    def predict_from_image(self, image_path: str, ref_image_path: str) -> Tuple[float, float]:
        """Return (pred_force_N, mean_depth_mm)."""
        self._ensure_reference(ref_image_path)
        self.cam.file_path = image_path
        bgr = self.cam.get_rectify_crop_image()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = simple_contact_mask(gray, threshold=5)
        mean_d, _, _, _ = compute_depth_stats(self.cam, bgr, mask)
        pred_force = self.intercept + self.coef_mean * mean_d
        return float(pred_force), float(mean_d)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Runtime force predictor using linear mean-depth model")
    parser.add_argument("--cfg", default="shape_reconstruction/shape_config.yaml")
    parser.add_argument("--sensor_id", type=int, default=3)
    parser.add_argument("--coef", default="force_estimation/analysis/force_regression_linear_mean.json")
    parser.add_argument("--ref", required=True, help="Path to reference (lowest-force) image")
    parser.add_argument("--img", required=True, help="Path to image to predict")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    pred = RuntimeForcePredictor(str(project_root / args.cfg), args.sensor_id, str(project_root / args.coef))
    img_path = str((project_root / args.img).resolve())
    ref_path = str((project_root / args.ref).resolve())
    fN, md = pred.predict_from_image(img_path, ref_path)
    print(f"Predicted force: {fN:.2f} N (mean depth {md:.3f} mm)")
