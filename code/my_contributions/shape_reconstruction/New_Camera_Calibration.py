import os
import numpy as np
import cv2
import yaml
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree


class ImagePreProcessing:
    def __init__(self, cfg, sample_image_path):
        """
        Initialize the image pre-processing with configuration settings.
        Load the reference and sample images, and pre-process them.
        Args:
            cfg (dict): Configuration dictionary loaded from YAML
            sample_image_path (str): Path to the sample image
        """
        self.cfg = cfg
        self.image_format = cfg['camera_calibration']['image_format']
        self.crop_size = tuple(cfg['camera_calibration']['crop_size'])
        self.sample_image_path = sample_image_path

        # Load the reference and sample images
        self.sample_image = self.load_image(self.sample_image_path)

        # Pre-process the reference and sample images
        self.sample_image = self.threshold_image(self.sample_image)

        # Detect the points on the reference and sample images
        self.sample_contours, self.contours_areas = self.detect_points(self.sample_image, debug=True)
        self.sample_contours, self.contours_areas = self.nms_on_centres(self.sample_contours, self.contours_areas, min_dist=8)
        # Draw the points on the reference and sample images
        #self.sample_image = self.draw_points(self.sample_image, self.sample_contours)
        
        # Draw the grid on the reference and sample images
        #self.sample_image = self.draw_grid(self.sample_image, self.sample_image_path)

    def load_image(self, image_path):
        """
        Load an image from the specified path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Loaded image in grayscale and denoised using Gaussian blur
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image at {image_path}")
        # Denoise the image
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return img
    
    def threshold_image(self, img):
        """
        Threshold the image using Otsu's method.
        
        Args:
            img (np.ndarray): Input image
            
        Returns:
            np.ndarray: Thresholded image
        """
        # Dynamically adjust threshold: stricter (higher) at edges, looser (lower) in center
        h, w = img.shape
        
        # Create a rectangular distance-from-center map, normalized to [0, 1]
        y_indices, x_indices = np.indices((h, w))
        cy, cx = h // 2, w // 2
        # Normalised offsets in each direction (0 at centre, 1 at the nearest edge)
        y_off = np.abs(y_indices - cy) / cy
        x_off = np.abs(x_indices - cx) / cx

        # Rectangular distance = max of the two offsets
        norm_dist = np.maximum(y_off, x_off)        # iso-lines are squares

        # Set thresholds: low in center (e.g. 60), high at edge (e.g. 40)
        center_thresh = 60
        edge_thresh = 40
        thresh_map = (1 - norm_dist) * center_thresh + norm_dist * edge_thresh
        thresh_map = thresh_map.astype(np.uint8)

        # Instead of binarizing to just black/white, create a 3-level image:
        # - dark (black) for pixels much darker than threshold
        # - medium (gray) for pixels near threshold
        # - light (white) for pixels much brighter than threshold
        lower_margin = 10 # pixels below (darker than) threshold by this margin are black
        upper_margin = 10 # pixels above (brighter than) threshold by this margin are white

        img_bin = np.full_like(img, 128, dtype=np.uint8)  # default to gray
        img_bin[img < (thresh_map - lower_margin)] = 0    # black for dark points
        img_bin[img > (thresh_map + upper_margin)] = 255  # white for bright points

        # Apply per-pixel threshold
        #img_bin = np.where(img < thresh_map, 0, 255).astype(np.uint8)
        return img_bin
    
    def detect_points(self, img, min_area: int = 50, min_circularity: float = 0.6, debug: bool = False):
        """
        Detect dark circular blobs in the thresholded image and return the filtered contours.

        The routine considers **only black pixels** (value 0 in the 3-level image),
        ignores very small regions and keeps only those whose shape is close to a
        circle so that their centroids can be computed reliably later.

        Args:
            img (np.ndarray): Thresholded image produced by ``threshold_image``.
            min_area (int):   Minimum contour area to keep.
            min_circularity (float): Minimum circularity (1.0 = perfect circle).
            debug (bool):     If True the accepted contours are drawn on *img*.

        Returns:
            list[np.ndarray]: List with the accepted contours.
        """
        # 1. Build a binary mask where the black pixels (0) become foreground (255)
        mask = np.zeros_like(img, dtype=np.uint8)
        mask[img == 0] = 255

        # 2. Find connected white blobs in that mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        good_contours = []
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue  # too small → skip

            # Circularity test:  4π·Area / Perimeter²  (1 = perfect circle)
            peri = cv2.arcLength(cnt, closed=True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri ** 2)
            if circularity < min_circularity:
                continue  # not round enough → skip

            good_contours.append(cnt)
            areas.append(area)
            
            if debug:
                cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)

        return good_contours, areas
    
    def nms_on_centres(self, contours, areas, min_dist=8):
        """
        Keep the largest contour in each neighbourhood.

        Args
        ----
        contours : list[np.ndarray]
            Contours returned by `detect_points`.
        areas    : list[int | float]
            Corresponding contour areas.
        min_dist : int
            Minimum pixel distance allowed between surviving centres.

        Returns
        -------
        kept_idx : list[int]
            Indices of the contours that survive the suppression.
        """
        # 1. extract the centroids once
        centres = np.array([
            (int(cv2.moments(c)['m01'] / (cv2.moments(c)['m00'] + 1e-6)),
            int(cv2.moments(c)['m10'] / (cv2.moments(c)['m00'] + 1e-6)))
            for c in contours
        ], dtype=np.float32)   # shape (N, 2)  ->  (row, col)

        # 2. sort by descending area
        order = np.argsort(-np.asarray(areas))
        centres_sorted = centres[order]

        # 3. KD-tree for neighbour queries
        tree = cKDTree(centres_sorted)

        keep = np.ones(len(contours), dtype=bool)          # in sorted order
        for i, centre in enumerate(centres_sorted):
            if not keep[i]:
                continue                                  # suppressed earlier
            # all points within radius, incl. itself
            neighbours = tree.query_ball_point(centre, r=min_dist)
            # drop every neighbour that comes *after* i in the ranking
            for j in neighbours:
                if j > i:
                    keep[j] = False

        kept_idx = order[keep]                            # back to original order
        contours_filt = [contours[i] for i in kept_idx]
        areas_filt    = [areas[i]    for i in kept_idx]
        return contours_filt, areas_filt

    def draw_points(self, img: np.ndarray, contours: list[np.ndarray]):
        """
        Draw the supplied *contours* on *img* and mark their centroids.

        Args:
            img (np.ndarray): Grayscale or colour image on which to draw.
            contours (list[np.ndarray]): Contours (e.g. output of ``detect_points``).

        Returns:
            tuple[np.ndarray, list[tuple[int,int]]]:
                1. Annotated image (same shape as *img*)
                2. List with centre coordinates (row, col) for each contour
        """
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        centres = []

        for cnt in contours:
            cv2.drawContours(annotated, [cnt], -1, (0, 0, 255), 2)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centres.append((cy, cx))
                cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), -1)

        return annotated, centres
    

        
        
        

class CalibrationFromImages:
    def __init__(self, cfg, ref_image_path, sample_image_path):
        """
        cfg: dict loaded from YAML
        ref_image_path: path to the flat reference PNG
        sample_image_path: path to the pressed sample PNG
        """
        self.ref_path = ref_image_path
        self.sample_path = sample_image_path

        calib = cfg['camera_calibration']
        self.row_points    = calib['row_points']
        self.col_points    = calib['col_points']
        self.grid_distance = calib['grid_distance']
        self.image_format  = calib['image_format']

        # derive output directory from ref image location
        self.out_dir = os.path.dirname(self.ref_path)
        os.makedirs(self.out_dir, exist_ok=True)

        # build paths for saving rectification maps
        root = cfg['calibration_root_dir']
        sid  = cfg['sensor_id']
        base = f"{root}/sensor_{sid}{calib['camera_calibration_dir']}"
        self.row_index_path      = f"{base}{calib['row_index_path']}"
        self.col_index_path      = f"{base}{calib['col_index_path']}"
        self.position_scale_path = f"{base}{calib['position_scale_path']}"

    def run(self):
        # (c) Load flat reference image
        ref = cv2.imread(self.ref_path, cv2.IMREAD_COLOR)
        if ref is None:
            raise FileNotFoundError(f"Cannot load reference image at {self.ref_path}")
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{self.out_dir}/ref_gray.{self.image_format}", ref_gray)

        # (c) Load pressed sample image
        sample = cv2.imread(self.sample_path, cv2.IMREAD_COLOR)
        if sample is None:
            raise FileNotFoundError(f"Cannot load sample image at {self.sample_path}")
        sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{self.out_dir}/sample_gray.{self.image_format}", sample_gray)

        # Ensure reference and sample have the same resolution
        if sample_gray.shape != ref_gray.shape:
            print(f"[Info] Resizing sample image from {sample_gray.shape} to match reference {ref_gray.shape}.")
            sample_gray = cv2.resize(sample_gray, (ref_gray.shape[1], ref_gray.shape[0]),
                                     interpolation=cv2.INTER_AREA)

        # Proceed with calibration on matching-size images
        self.calibrate_image(ref_gray, sample_gray)

    def calibrate_image(self, ref, sample):
        # Compute difference and mask
        diff = cv2.subtract(ref, sample)
        mask = (diff < 100).astype(np.uint8)
        diff = diff * mask
        diff[diff < 5] = 0

        # Adaptive threshold and morphology
        binary = cv2.adaptiveThreshold(diff, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 51, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # (d) Detect calibration board imprints
        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 2000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    pts.append([cy, cx])
        pts = np.array(pts)

        expected = self.row_points * self.col_points
        if pts.size == 0 or pts.shape[0] != expected:
            # Visualize detected points for debugging purposes
            debug_img = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
            for (y, x) in pts:
                cv2.circle(debug_img, (x, y), 4, (0, 0, 255), -1)

            cv2.imshow('Detected Dots (Debug)', debug_img)
            cv2.imwrite(f"{self.out_dir}/detected_dots_debug.{self.image_format}", debug_img)
            cv2.waitKey(0)
            cv2.destroyWindow('Detected Dots (Debug)')

            raise ValueError(
                f"Detected {pts.shape[0]} calibration dots but expected {expected}. "
                "Displayed debug image with detected dots; adjust thresholds or improve sample image quality.")

        # Sort points into grid
        pts = pts[np.lexsort(pts[:, ::-1].T)]
        for i in range(self.row_points):
            row = pts[i*self.col_points:(i+1)*self.col_points]
            pts[i*self.col_points:(i+1)*self.col_points] = row[np.lexsort(row.T)]

        # Compute average distance around center
        center_idx = expected // 2
        neighbors  = [-self.col_points, -1, 1, self.col_points]
        dist_sum   = sum(np.linalg.norm(pts[center_idx] - pts[center_idx+o]) for o in neighbors)
        dist_avg   = dist_sum / 4

        # Save pixel-to-mm scale and center position
        pos_scale = pts[center_idx].tolist()
        pos_scale.append(self.grid_distance / dist_avg)
        np.save(self.position_scale_path, pos_scale)

        # Prepare RBF remapping
        real = np.zeros_like(pts, dtype=float)
        for i in range(self.row_points):
            for j in range(self.col_points):
                idx = i*self.col_points + j
                real[idx] = (
                    pts[center_idx] + dist_avg * np.array([
                        i - self.row_points//2,
                        j - self.col_points//2
                    ])
                )
        itp_row = Rbf(real[:,0], real[:,1], pts[:,0], function='cubic')
        itp_col = Rbf(real[:,0], real[:,1], pts[:,1], function='cubic')

        h, w = ref.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        row_idx = np.clip(itp_row(gy, gx).astype(int), 0, h-1)
        col_idx = np.clip(itp_col(gy, gx).astype(int), 0, w-1)

        # Save rectification maps
        np.save(self.row_index_path, row_idx)
        np.save(self.col_index_path, col_idx)

        # (e) Rectify & crop the pressed imprint
        rectified = sample[row_idx, col_idx]
        # Display rectified grid imprint
        cv2.imshow('Rectified Imprint', rectified)
        cv2.waitKey(0)


if __name__ == '__main__':
    # Load configuration
    with open('shape_config.yaml', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Paths to your PNGs
    #ref_png    = 'calibration/sensor_2/camera_calibration/ref.png'
    sample_png = 'calibration/sensor_3/camera_calibration/sample9x7_raw.jpg'
    
    #print(f"Size of sample image: {cv2.imread(sample_png).shape}")
    #cv2.imshow('Reference Image', cv2.imread(ref_png))
    
    """cv2.imshow('Sample Image', cv2.imread(sample_png))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    processed_image = ImagePreProcessing(cfg, sample_png)
    
    #show the reference and sample images
    #cv2.imshow('Reference Image', processed_image.ref_image)
    cv2.imshow('Sample Image', processed_image.sample_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pre = ImagePreProcessing(cfg, sample_png)
    good_cnts, areas = pre.detect_points(pre.sample_image, min_area=380,
                                min_circularity=0.05, debug=True)
    good_cnts, areas = pre.nms_on_centres(good_cnts, areas, min_dist=100)
    print(f"Detected {len(areas)} areas: {areas}")
    annotated_img, centres = pre.draw_points(pre.sample_image.copy(), good_cnts)

    # ─── Save documentation images ─────────────────────────────────────
    save_dir = os.path.dirname(sample_png)
    gray_img = cv2.imread(sample_png, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(save_dir, 'sample_gray.png'), gray_img)
    cv2.imwrite(os.path.join(save_dir, 'sample_contours.png'), pre.sample_image)
    cv2.imwrite(os.path.join(save_dir, 'sample_contours_points.png'), annotated_img)

    cv2.imshow('centres', annotated_img)
    print(f"Detected {len(centres)} centres: {centres}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #save the reference and sample images
    #cv2.imwrite('reference_image.png', processed_image.ref_image)
    cv2.imwrite('sample_image.png', processed_image.sample_image)
    
if __name__ == '__main__':
    """
    Stand-alone calibration routine that:

      2. detects the pin-tip imprints on the gel surface,
      3. builds the distortion-correction maps (row_index / col_index) and
         the position-scale file (centre position + pixel/mm),
      4. shows the rectified image for visual confirmation.

    The implementation follows the same idea as `_1_Camera_Calibration.py`
    but relies solely on the detected points – no flat reference image is
    required.
    """
    # ─── 1. Configuration ──────────────────────────────────────────────
    with open('shape_config.yaml', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # TODO: adjust to your actual sample path
    sample_png = 'calibration/sensor_3/camera_calibration/sample9x7_raw.jpg'

    # ─── 2. Point detection ────────────────────────────────────────────
    pre = ImagePreProcessing(cfg, sample_png)
    good_cnts, areas = pre.detect_points(pre.sample_image,
                                         min_area=380,
                                         min_circularity=0.05,
                                         debug=True)
    good_cnts, areas = pre.nms_on_centres(good_cnts, areas, min_dist=100)
    _, centres = pre.draw_points(pre.sample_image.copy(), good_cnts)

    print(f"Detected {len(centres)} centres.")

    # ─── 3. Build and save rectification maps ──────────────────────────
    def build_rectification(centres_list, cfg, img_shape):
        """
        Given the ordered list of (row, col) centres and the image shape
        generate and save the rectification maps + position_scale.
        """
        cam_cal = cfg['camera_calibration']
        row_pts = cam_cal['row_points']
        col_pts = cam_cal['col_points']
        grid_dist = cam_cal['grid_distance']

        if len(centres_list) != row_pts * col_pts:
            raise ValueError(f"Expected {row_pts*col_pts} points, got {len(centres_list)}")

        pts = np.asarray(centres_list, dtype=float)

        # sort by row (y) then column (x)
        pts = pts[np.lexsort(pts[:, ::-1].T)]
        for i in range(row_pts):
            row_slice = pts[i*col_pts:(i+1)*col_pts]
            pts[i*col_pts:(i+1)*col_pts] = row_slice[np.lexsort(row_slice.T)]

        center_idx = (row_pts * col_pts)//2
        neigh = [-col_pts, -1, 1, col_pts]
        dist_avg = sum(np.linalg.norm(pts[center_idx] - pts[center_idx+n]) for n in neigh) / 4

        # Build calibration directory paths
        sensor_id = cfg['sensor_id']
        root = cfg['calibration_root_dir']
        cal_dir = cam_cal['camera_calibration_dir']
        base = f"{root}/sensor_{sensor_id}{cal_dir}"
        os.makedirs(base, exist_ok=True)

        row_index_path = base + cam_cal['row_index_path']
        col_index_path = base + cam_cal['col_index_path']
        position_scale_path = base + cam_cal['position_scale_path']

        # centre position + pixel per mm
        pos_scale = pts[center_idx].tolist()
        pos_scale.append(grid_dist / dist_avg)
        np.save(position_scale_path, pos_scale)

        # Expected ideal grid (real world)
        real = np.zeros_like(pts, dtype=float)
        for i in range(row_pts):
            for j in range(col_pts):
                idx = i*col_pts + j
                real[idx] = pts[center_idx] + dist_avg * np.array([i - row_pts//2,
                                                                    j - col_pts//2])

        itp_row = Rbf(real[:,0], real[:,1], pts[:,0], function='cubic')
        itp_col = Rbf(real[:,0], real[:,1], pts[:,1], function='cubic')

        h, w = img_shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        row_index = np.clip(itp_row(gy, gx).astype(int), 0, h-1)
        col_index = np.clip(itp_col(gy, gx).astype(int), 0, w-1)

        np.save(row_index_path, row_index)
        np.save(col_index_path, col_index)

        return row_index, col_index

    # Load original colour sample for visualisation / rectification
    sample_color = cv2.imread(sample_png, cv2.IMREAD_COLOR)
    if sample_color is None:
        raise FileNotFoundError(f"Could not load {sample_png}")

    row_map, col_map = build_rectification(centres, cfg, sample_color.shape[:2])

    # ─── 4. Visual check ───────────────────────────────────────────────
    rectified = sample_color[row_map, col_map]
    cv2.imwrite(os.path.join(save_dir, 'sample_rectified.png'), rectified)
    cv2.imshow('Rectified Imprint', rectified)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if False:
    """
    Stand-alone calibration routine that:

      1. loads the raw sample image,
      2. detects the pin-tip imprints on the gel surface,
      3. builds the distortion-correction maps (row_index / col_index) and
         the position-scale file (centre position + pixel/mm),
      4. shows the rectified image for visual confirmation.

    The implementation follows the same idea as `_1_Camera_Calibration.py`
    but relies solely on the detected points – no flat reference image is
    required.
    """
    # ─── 1. Configuration ──────────────────────────────────────────────
    with open('shape_config.yaml', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # TODO: adjust to your actual sample path
    sample_png = 'calibration/sensor_3/camera_calibration/sample9x7_raw.jpg'

    # ─── 2. Point detection ────────────────────────────────────────────
    pre = ImagePreProcessing(cfg, sample_png)
    good_cnts, areas = pre.detect_points(pre.sample_image,
                                         min_area=380,
                                         min_circularity=0.05,
                                         debug=True)
    good_cnts, areas = pre.nms_on_centres(good_cnts, areas, min_dist=100)
    _, centres = pre.draw_points(pre.sample_image.copy(), good_cnts)

    print(f"Detected {len(centres)} centres.")

    # ─── 3. Build and save rectification maps ──────────────────────────
    def build_rectification(centres_list, cfg, img_shape):
        """
        Given the ordered list of (row, col) centres and the image shape
        generate and save the rectification maps + position_scale.
        """
        cam_cal = cfg['camera_calibration']
        row_pts = cam_cal['row_points']
        col_pts = cam_cal['col_points']
        grid_dist = cam_cal['grid_distance']

        if len(centres_list) != row_pts * col_pts:
            raise ValueError(f"Expected {row_pts*col_pts} points, got {len(centres_list)}")

        pts = np.asarray(centres_list, dtype=float)

        # sort by row (y) then column (x)
        pts = pts[np.lexsort(pts[:, ::-1].T)]
        for i in range(row_pts):
            row_slice = pts[i*col_pts:(i+1)*col_pts]
            pts[i*col_pts:(i+1)*col_pts] = row_slice[np.lexsort(row_slice.T)]

        center_idx = (row_pts * col_pts)//2
        neigh = [-col_pts, -1, 1, col_pts]
        dist_avg = sum(np.linalg.norm(pts[center_idx] - pts[center_idx+n]) for n in neigh) / 4

        # Build calibration directory paths
        sensor_id = cfg['sensor_id']
        root = cfg['calibration_root_dir']
        cal_dir = cam_cal['camera_calibration_dir']
        base = f"{root}/sensor_{sensor_id}{cal_dir}"
        os.makedirs(base, exist_ok=True)

        row_index_path = base + cam_cal['row_index_path']
        col_index_path = base + cam_cal['col_index_path']
        position_scale_path = base + cam_cal['position_scale_path']

        # centre position + pixel per mm
        pos_scale = pts[center_idx].tolist()
        pos_scale.append(grid_dist / dist_avg)
        np.save(position_scale_path, pos_scale)

        # Expected ideal grid (real world)
        real = np.zeros_like(pts, dtype=float)
        for i in range(row_pts):
            for j in range(col_pts):
                idx = i*col_pts + j
                real[idx] = pts[center_idx] + dist_avg * np.array([i - row_pts//2,
                                                                    j - col_pts//2])

        itp_row = Rbf(real[:,0], real[:,1], pts[:,0], function='cubic')
        itp_col = Rbf(real[:,0], real[:,1], pts[:,1], function='cubic')

        h, w = img_shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        row_index = np.clip(itp_row(gy, gx).astype(int), 0, h-1)
        col_index = np.clip(itp_col(gy, gx).astype(int), 0, w-1)

        np.save(row_index_path, row_index)
        np.save(col_index_path, col_index)

        return row_index, col_index

    # Load original colour sample for visualisation / rectification
    sample_color = cv2.imread(sample_png, cv2.IMREAD_COLOR)
    if sample_color is None:
        raise FileNotFoundError(f"Could not load {sample_png}")

    row_map, col_map = build_rectification(centres, cfg, sample_color.shape[:2])

    # ─── 4. Visual check ───────────────────────────────────────────────
    rectified = sample_color[row_map, col_map]
    cv2.imshow('Rectified Imprint', rectified)
    cv2.waitKey(0)
    cv2.destroyAllWindows()