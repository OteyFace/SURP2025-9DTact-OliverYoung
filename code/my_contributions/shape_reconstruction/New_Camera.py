import numpy as np
import cv2


class Camera:
    def __init__(self, cfg, calibrated=True, file_mode=False, file_path=None):
        """
        cfg: dict loaded from your YAML, must include:
           - sensor_id
           - calibration_root_dir
           - camera_setting: {camera_channel, resolution, fps}
           - camera_calibration: {camera_calibration_dir, row_index_path,
                                 col_index_path, position_scale_path,
                                 crop_size}
        calibrated: if True, loads row/col index & position_scale
        file_mode: if True, skips VideoCapture and always load from file_path
        file_path: path to .png/.jpg when file_mode=True
        """
        self.file_mode = file_mode
        self.file_path = file_path

        # ─── SETUP CAPTURE ────────────────────────────────────────────
        if not self.file_mode:
            cam_cfg = cfg['camera_setting']
            self.cap = cv2.VideoCapture(cam_cfg['camera_channel'])
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera #{cam_cfg['camera_channel']}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_cfg['resolution'][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg['resolution'][1])
            self.cap.set(cv2.CAP_PROP_FPS,          cam_cfg['fps'])
        else:
            self.cap = None

        # ─── CALIBRATION FILEPATHS ───────────────────────────────────
        sensor_id = cfg['sensor_id']
        root   = cfg['calibration_root_dir']
        cal    = cfg['camera_calibration']
        # define the base calibration directory path
        self.camera_calibration_dir = f"{root}/sensor_{sensor_id}{cal['camera_calibration_dir']}"
        # then build all dependent file paths
        self.row_index_path      = f"{self.camera_calibration_dir}{cal['row_index_path']}"
        self.col_index_path      = f"{self.camera_calibration_dir}{cal['col_index_path']}"
        self.position_scale_path = f"{self.camera_calibration_dir}{cal['position_scale_path']}"
        self.crop_img_height     = cal['crop_size'][0]
        self.crop_img_width      = cal['crop_size'][1]

        # ─── LOAD CALIBRATION MAPS ───────────────────────────────────
        if calibrated:
            self.row_index     = np.load(self.row_index_path)
            self.col_index     = np.load(self.col_index_path)
            position_scale     = np.load(self.position_scale_path)
            center_position    = position_scale[0:2]
            self.pixel_per_mm  = position_scale[2]
            self.height_begin  = int(center_position[0] - self.crop_img_height/2)
            self.height_end    = int(center_position[0] + self.crop_img_height/2)
            self.width_begin   = int(center_position[1] - self.crop_img_width/2)
            self.width_end     = int(center_position[1] + self.crop_img_width/2)

    def get_raw_image(self):
        """Return a single BGR image, either from file or camera."""
        if self.file_mode:
            img = cv2.imread(self.file_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not load image at {self.file_path}")
            return img
        else:
            ret, img = self.cap.read()
            if not ret or img is None:
                raise RuntimeError("Failed to grab frame from camera")
            return img

    def _match_calibration_resolution(self, img):
        """Resize *img* to the resolution expected by the calibration maps.

        The pre-computed ``row_index`` / ``col_index`` arrays describe the
        sampling pattern for an image of shape  *(H,W,3)*  where
        ``(H,W) = row_index.shape``.  If a file loaded in *file_mode* has a
        different resolution we first rescale it so the rectification indices
        remain valid.
        """
        exp_h, exp_w = self.row_index.shape
        if img.shape[0] != exp_h or img.shape[1] != exp_w:
            # Resize to expected width × height (cv2 expects (width,height) order)
            img = cv2.resize(img, (exp_w, exp_h), interpolation=cv2.INTER_AREA)
        return img

    def rectify_image(self, img):
        """Apply distortion correction using precomputed row/col indices.

        Automatically resizes the input if its resolution differs from the
        calibration maps (useful for pre-recorded frames with a lower
        resolution).
        """
        img = self._match_calibration_resolution(img)
        return img[self.row_index, self.col_index]

    def crop_image(self, img):
        """Crop to the calibrated ROI centered around the gel."""
        return img[self.height_begin:self.height_end,
                   self.width_begin:self.width_end]

    def get_rectify_image(self):
        return self.rectify_image(self.get_raw_image())

    def get_rectify_crop_image(self):
        return self.crop_image(self.get_rectify_image())

    def get_raw_avg_image(self, n_frames=10):
        """
        For camera: wait for ‘y’, then average n_frames.
        For file_mode: just load & return the file.
        """
        if self.file_mode:
            return self.get_raw_image()

        # else: interactive capture
        while True:
            frame = self.get_raw_image()
            cv2.imshow('Press y to capture reference', frame)
            if cv2.waitKey(1) == ord('y'):
                cv2.destroyAllWindows()
                break

        # average next n_frames
        acc = np.zeros_like(frame, dtype=np.float64)
        for _ in range(n_frames):
            acc += self.get_raw_image().astype(np.float64)
        return (acc / n_frames).astype(np.uint8)

    def get_rectify_avg_image(self, n_frames=10):
        """Same as get_raw_avg_image but produces undistorted results."""
        if self.file_mode:
            return self.get_rectify_image()

        while True:
            frame = self.get_rectify_image()
            cv2.imshow('Press y to capture reference', frame)
            if cv2.waitKey(1) == ord('y'):
                cv2.destroyAllWindows()
                break

        acc = np.zeros_like(frame, dtype=np.float64)
        for _ in range(n_frames):
            acc += self.get_rectify_image().astype(np.float64)
        return (acc / n_frames).astype(np.uint8)

    def get_rectify_crop_avg_image(self, n_frames=10):
        """Undistort + crop + average."""
        if self.file_mode:
            return self.get_rectify_crop_image()

        while True:
            frame = self.get_rectify_crop_image()
            cv2.imshow('Press y to capture reference', frame)
            if cv2.waitKey(1) == ord('y'):
                cv2.destroyAllWindows()
                break

        acc = np.zeros_like(frame, dtype=np.float64)
        for _ in range(n_frames):
            acc += self.get_rectify_crop_image().astype(np.float64)
        return (acc / n_frames).astype(np.uint8)

    def img_list_avg_rectify(self, img_list):
        """
        Given a list of file paths, load each, sum, average, then rectify.
        """
        first = cv2.imread(img_list[0], cv2.IMREAD_COLOR)
        acc = np.zeros_like(first, dtype=np.float64)
        for p in img_list:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not load image at {p}")
            acc += img.astype(np.float64)
        avg = (acc / len(img_list)).astype(np.uint8)
        return self.rectify_image(avg)


# Optional demonstration
if __name__ == '__main__':
    import yaml

    # Load your config
    with open("shape_config.yaml", encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Demo in file_mode
    demo_path = "9DTact/shape_reconstruction/calibration/sensor_3/camera_calibration/sample_rectified.png"
    cam = Camera(cfg, calibrated=True, file_mode=True, file_path=demo_path)

    img = cam.get_raw_avg_image()
    cv2.imshow('Demo Load', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
