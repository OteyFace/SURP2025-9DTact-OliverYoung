"""Contact Region Detection without reference image.

This script loads a single *sample* frame, rectifies & crops it using the
previously-computed calibration maps, then attempts to segment the regions
where the gel is in contact with an object.

Algorithm overview
------------------
1.  Rectify + crop using the calibration data (same code path as *Camera*).
2.  Convert to grayscale and remove glare / noise with a Gaussian blur.
3.  Estimate the background illumination by a large-kernel Gaussian blur and
    subtract it from the image ("high-pass" / top-hat).  This highlights local
    darker / brighter spots caused by contact.
4.  Threshold the high-frequency component (magnitude) and clean the mask with
    morphological open/close to produce the contact region mask.

The resulting mask is displayed together with the input image for
inspection.

Run from *shape_reconstruction* directory, for instance:
    python New_Contact_Region_Detection.py \
        --img ..\\calibration\\frames\\Screw_1\\00237.jpg 

The default config file (`shape_config.yaml`) is used; override the
`--sensor_id` if you calibrated a different sensor.
"""

import argparse
import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

try:
    # when imported as package: shape_reconstruction.New_Camera
    from .New_Camera import Camera  # type: ignore
except Exception:  # fallback when running this file directly
    from New_Camera import Camera

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from PIL import Image

# Note: removed unused ContactDetector import to avoid linter warning

# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def load_cfg(path: str, sensor_id: int):
    """Load YAML configuration and override the sensor id if supplied."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['sensor_id'] = sensor_id
    return cfg


def rectify_and_crop(img_path: str, cfg: dict):
    """Return the rectified & cropped BGR image for *img_path* using *cfg*."""
    cam = Camera(cfg, calibrated=True, file_mode=True, file_path=img_path)
    return cam.get_rectify_crop_image()


def detect_contact_region(bgr, gray: np.ndarray,
                          hp_blur: int = 101,
                          threshold: int = 10,
                          morph_ks: int = 5):  # lowered threshold for higher sensitivity
    """Create contact region mask from cropped grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Cropped grayscale input (uint8).
    hp_blur : int
        Kernel size (odd) for Gaussian blur used as background estimate.
    threshold : int
        Minimum absolute difference from background to classify a pixel as
        contact.
    morph_ks : int
        Kernel size for morphological open/close operations.
    """
    
    # 1. Estimate background (low-frequency illumination)
    bg = cv2.GaussianBlur(gray, (hp_blur, hp_blur), 0)

    # 2. High-frequency / detail image (signed)
    diff = gray.astype(np.int16) - bg.astype(np.int16)
    
    # 3. Magnitude of deviation
    mag = np.abs(diff).astype(np.uint8)

    # 4. Threshold – pixels with |diff| > threshold are considered contact. If
    # *threshold* <= 0 we fall back to Otsu automatic thresholding which often
    # works better when lighting varies.
    if threshold <= 0:
        # Otsu will pick a value; make sure mag is not all zeros
        if np.count_nonzero(mag) == 0:
            mask = np.zeros_like(mag)
        else:
            _, mask = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)

    # 5. Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks, morph_ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    
    # post-processing of contact mask using SAM-2 segmentation
    
    # 1. SAM2 segmentation, isolate contact objects and remove all other segments
    segs = segment_image_sam2(bgr)
    
    # ─── SAM-2 segmentation ─────────────────────────────────────
    
    # If object segment alligns with contact region, use raw image behind segments  
    contacting_segments = []
    for ann in segs:
        seg_mask = ann["segmentation"]  # binary mask
        # Identify segments that overlap with the contact mask
        if seg_mask.shape != mask.shape:
            seg_mask = cv2.resize(seg_mask.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        if np.any(seg_mask & (mask > 0)):
            contacting_segments.append(ann)
    
    # Make mask be the raw rgb image containing the contact region, mask does not need to be binary
    mask = np.zeros_like(gray, dtype=np.uint8)  # start with an empty mask     
    if contacting_segments:
        # If there are segments in contact, combine their masks
        for ann in contacting_segments:
            seg_mask = ann["segmentation"]
            if seg_mask.shape != mask.shape:
                seg_mask = cv2.resize(seg_mask.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask[seg_mask] = 255
    return mask


def isolate_contact_area(bgr: np.ndarray, mask: np.ndarray):
    """Extract only the contact area from the raw image using the mask.
    
    Parameters
    ----------
    bgr : np.ndarray
        Raw BGR image (rectified and cropped).
    mask : np.ndarray
        Binary mask where contact regions are white (255).
        
    Returns
    -------
    np.ndarray
        Image containing only the contact area, with non-contact pixels set to black.
    """
    # Create a copy of the original image
    contact_isolated = bgr.copy()
    
    # Set all pixels outside the contact mask to black
    contact_isolated[mask == 0] = 0
    
    return contact_isolated


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255)):
    """Overlay *mask* on *bgr* image (red by default).
    
    Take raw image and subtract all pixels that are outside the segment mask
    """
    overlay = bgr.copy()
    overlay[mask > 0] = color  # set mask pixels to red
    
    # subtract all pixels that are outside the segment mask
    overlay[mask == 0] = 0  # set pixels outside the segment mask to black
    return overlay

# ──────────────────────────────────────────────────────────────────────────────
# SAM-2 Segmentation
# ──────────────────────────────────────────────────────────────────────────────

def segment_image_sam2(bgr: np.ndarray, device: str = "cpu"):
    """Run SAM-2 automatic segmentation…"""
    
    # first call: cache the generator
    if not hasattr(segment_image_sam2, "_generator"):
        # root → …/9DTact/sam2
        root = "C:/Users/ojyca/OneDrive/Documents/SURP 2025/sam2"
        # build absolute paths
        ckpt_path = os.path.join(root, "sam2.1_hiera_large.pt")
        cfg_path = os.path.join(root, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")

        # debug: print to verify
        print(f"[DEBUG] SAM2 ckpt at {ckpt_path} →", os.path.isfile(ckpt_path))
        print(f"[DEBUG] SAM2 cfg  at {cfg_path} →", os.path.isfile(cfg_path))

        if not (os.path.isfile(ckpt_path) and os.path.isfile(cfg_path)):
            print("[WARN] SAM-2 checkpoint or config not found – segmentation skipped.")
            segment_image_sam2._generator = None
        else:
            print("[INFO] Loading SAM-2 model from local checkpoint …")
            sam2_model = build_sam2(cfg_path, ckpt_path, device=device, apply_postprocessing=False)
            segment_image_sam2._generator = SAM2AutomaticMaskGenerator(
                sam2_model,
                points_per_side=32,
                points_per_batch=64,
                output_mode="binary_mask",
            )

    generator = segment_image_sam2._generator
    if generator is None:
        return []

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return generator.generate(rgb)


# ──────────────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Single-image contact region detection')
    parser.add_argument('--img', type=str,
                        default='calibration/frames/Screw_1/00237.jpg',
                        help='Path to sample frame')
    parser.add_argument('--cfg', type=str, default='shape_config.yaml',
                        help='Calibration / sensor configuration YAML')
    parser.add_argument('--sensor_id', type=int, default=3,
                        help='Sensor id (matches calibration folder)')
    parser.add_argument('--show', action='store_true', help='Show intermediate steps')
    parser.add_argument('--use_sam', action='store_true', help='Run SAM-2 segmentation to locate contacting objects')
    parser.add_argument('--save_sam', action='store_true', help='Save a colour overlay of raw SAM-2 segments')
    parser.add_argument('--show_sam', action='store_true', help='Display the raw SAM-2 segment overlay in a window')

    # Detection tunables
    parser.add_argument('--threshold', type=int, default=5,
                        help='Absolute intensity-difference threshold. 0 or negative → Otsu adaptive.')
    parser.add_argument('--hp_blur', type=int, default=51,
                        help='Gaussian kernel (odd) used for background estimation.')
    parser.add_argument('--morph_ks', type=int, default=5,
                        help='Morphological kernel size.')
    args = parser.parse_args()

    if not os.path.isfile(args.img):
        raise FileNotFoundError(f"Image not found: {args.img}")
    if False:
        cfg = load_cfg(args.cfg, args.sensor_id)

        # ─── Rectify & crop ───────────────────────────────────────────────────
        bgr = rectify_and_crop(args.img, cfg)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ─── Contact detection ───────────────────────────────────────────────
        
        mask = detect_contact_region(bgr, gray,
                                    hp_blur=args.hp_blur,
                                    threshold=args.threshold,
                                    morph_ks=args.morph_ks)

        # ─── Visualisation & Saving ─────────────────────────────────────────
        overlay = overlay_mask(bgr, mask)
        contact_isolated = isolate_contact_area(bgr, mask)

        # Prepare output directory & file prefix
        out_dir = os.path.join("calibration", "images")
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.splitext(os.path.basename(args.img))[0]

        # Save core images
        # save images iteratevly to avoid overwriting
        #if out_dir exsist, add a number to the beginning of the file name
        if os.path.exists(out_dir):
            i = 1
            while os.path.exists(os.path.join(out_dir, f"{i}_{prefix}.png")):
                i += 1
            prefix = f"{i}_{prefix}"

        cv2.imwrite(os.path.join(out_dir, f"{prefix}_rect_crop.png"), bgr)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_contact_mask.png"), mask)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_overlay.png"), overlay)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_contact_isolated.png"), contact_isolated)

        # ─── Optional: visualise / save raw SAM-2 segmentation ─────────────────────
        if args.use_sam:
            segs = segment_image_sam2(bgr)
            # ─── SAM-2 segmentation ─────────────────────────────────────
            
            # If object segment alligns with contact region, use raw image behind segments  
            contacting_segments = []
            for ann in segs:
                seg_mask = ann["segmentation"]  # binary mask
                # Identify segments that overlap with the contact mask
                if seg_mask.shape != mask.shape:
                    seg_mask = cv2.resize(seg_mask.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                if np.any(seg_mask & (mask > 0)):
                    contacting_segments.append(ann)
            
            # Make mask be the raw rgb image containing the contact region, mask does not need to be binary
            mask = np.zeros_like(gray, dtype=np.uint8)  # start with an empty mask     
            if contacting_segments:
                # If there are segments in contact, combine their masks
                for ann in contacting_segments:
                    seg_mask = ann["segmentation"]
                    if seg_mask.shape != mask.shape:
                        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    mask[seg_mask] = 255

            # ─── Optional: visualise / save raw SAM-2 segmentation ─────────────────────
            if segs and (args.save_sam or args.show_sam):
                colour_overlay = bgr.copy()
                np.random.seed(0)
                for ann in segs:
                    colour = np.random.randint(0, 255, size=3).tolist()
                    seg_mask = ann["segmentation"]
                    if seg_mask.shape != colour_overlay.shape[:2]:
                        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (colour_overlay.shape[1], colour_overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    colour_overlay[seg_mask] = colour

                if args.save_sam:
                    cv2.imwrite(os.path.join(out_dir, f"{prefix}_sam2_segments.png"), colour_overlay)
                if args.show_sam:
                    cv2.imshow('SAM-2 raw segments', colour_overlay)

        if False:
            # Colour contacting segments with random colours
            if contacting_segments:
                colour_overlay = bgr.copy()
                np.random.seed(0)
                for ann in contacting_segments:
                    colour = np.random.randint(0, 255, size=3).tolist()
                    seg_mask = ann["segmentation"]
                    if seg_mask.shape != colour_overlay.shape[:2]:
                        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (colour_overlay.shape[1], colour_overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    colour_overlay[seg_mask] = colour
                cv2.imshow('Contact Objects (SAM-2)', colour_overlay)
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_sam2_objects.png"), colour_overlay)

                # ── Contours & grayscale depth-ready image ──────────────────
                combined_mask = np.zeros(mask.shape, dtype=np.uint8)
                for ann in contacting_segments:
                    seg = ann["segmentation"]
                    if seg.shape != combined_mask.shape:
                        seg = cv2.resize(seg.astype(np.uint8), (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    combined_mask[seg] = 255
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_vis = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_vis, contours, -1, (255, 255, 255), 2)
                gray_contour = cv2.cvtColor(contour_vis, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Contact Contours (gray)', gray_contour)
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_contours_gray.png"), gray_contour)
                print(f"SAM-2 detected {len(segs)} segments; {len(contacting_segments)} are in contact.")
            else:
                if args.use_sam:
                    print("SAM-2 found no segment overlapping contact mask.")

    sample9x7_raw = "calibration/sensor_3/camera_calibration/sample9x7_raw.jpg"
    sample9x7_segmented = "calibration/sensor_3/camera_calibration/sample9x7_segmented.png"
    sample_GRAY = "calibration/sensor_3/camera_calibration/sample_GRAY.png"
    sample_contours = "calibration/sensor_3/camera_calibration/sample_contours.png"
    sample_contours_points = "calibration/sensor_3/camera_calibration/sample_contours_points.png"
    sample_rectified ="calibration/sensor_3/camera_calibration/sample_rectified_crop.png"
    
    
    """
    Visualize the contact detection results.
    
    Args:
        contact_mask: Binary mask of detected contact areas
        contours: List of contact contours
        centers: List of contact center points
        save_path (str): Optional path to save the visualization
    """
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # img as np arrays
    sample9x7_raw_img = np.asarray(Image.open(sample9x7_raw))
    sample9x7_segmented_img = np.asarray(Image.open(sample9x7_segmented))
    sample_GRAY_img = np.asarray(Image.open(sample_GRAY))
    sample_contours_img = np.asarray(Image.open(sample_contours))
    sample_contours_points_img = np.asarray(Image.open(sample_contours_points))
    sample_rectified_img = np.asarray(Image.open(sample_rectified))
    
    
    # Original images
    axes[0, 0].imshow(sample9x7_raw_img)
    axes[0, 0].set_title('Raw Image')
    axes[0, 0].axis('off')
    
    # Grayscale images
    '''
    axes[0, 1].imshow(sample_GRAY_img, cmap='gray')
    axes[0, 1].set_title('Raw Image (Grayscale)')
    axes[0, 1].axis('off')
    '''
    sample_pointmap_OG = "calibration\sensor_1\camera_calibration\sample_drawing.png"
    sample_pointmap_OG = np.asarray(Image.open(sample_pointmap_OG))
    
    axes[0, 1].imshow(sample_pointmap_OG)
    axes[0, 1].set_title('Original 9DTact Pointmap')
    axes[0, 1].axis('off')
    
    # Points and Contours (Removed)
    '''
    axes[0, 2].imshow(sample_contours_img, cmap='gray')
    axes[0, 2].set_title('Raw Image Contours')
    axes[0, 2].axis('off')
    '''
    # Comparison with OG 9DTact
    sample_contour_OG = "calibration\sensor_1\camera_calibration\sample_contours.png"
    sample_contour_OG = np.asarray(Image.open(sample_contour_OG))
    
    axes[0, 2].imshow(sample_contour_OG, cmap='gray')
    axes[0, 2].set_title('Raw Image Contours')
    axes[0, 2].axis('off')
    
    
    axes[1, 0].imshow(sample_contours_points_img)
    axes[1, 0].set_title('Point Map Generated from Contours')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sample_rectified_img)
    axes[1, 1].set_title('Rectification Script Output (Removed lense distortion)')
    axes[1, 1].axis('off')
    # SAM2 (Remove)
    '''
    axes[1, 2].imshow(sample9x7_segmented_img)
    axes[1, 2].set_title('SAM2 Segmentation of Raw Image ')
    axes[1, 2].axis('off')
    '''
    # Comparison with OG 9DTact
    sample_cropped_OG = "calibration\sensor_1\camera_calibration\sample_new_crop.png"
    sample_cropped_OG = np.asarray(Image.open(sample_cropped_OG))
    
    axes[1, 2].imshow(sample_cropped_OG, cmap='gray')
    axes[1, 2].set_title('Original 9DTact Rectified and Cropped')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    contact_detection_save_path="contact_detection_results.png"
    
    plt.savefig(contact_detection_save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    cv2.imshow('Rectified + Cropped', bgr)
    cv2.imshow('Contact Mask', mask)
    cv2.imshow('Overlay', overlay)
    cv2.imshow('Contact Area Isolated', contact_isolated)
    if args.show:
        # Show the high-frequency magnitude for debugging
        bg = cv2.GaussianBlur(gray, (51, 51), 0)
        mag = cv2.convertScaleAbs(gray.astype(np.int16) - bg.astype(np.int16))
        cv2.imshow('High-freq magnitude', mag)
    print("Press any key (or close windows) to exit …")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
