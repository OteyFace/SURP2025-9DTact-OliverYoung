"""
 *
 * @file    image_data_segmentation.py
 * @edited by:  Oliver Young
 * @author:  ojyoung (ojyoung@ucsc.edu)
 * @date:  July 3, 2025
 *
 * This file contains original code utilizing the SAM2 segmentation model by Meta AI in conjuction with the 9DTact sensor library.
 * The class loads image data from a directory and uses it to segment images.       
 *
 * References:
 * https://github.com/facebookresearch/sam2
 * https://github.com/linchangyi1/9DTact
 *
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import yaml
import torch
import torchvision

from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

"""
 *
 * @file    calibrated_contact_detection.py
 * @edited by:  Oliver Young
 * @author:  ojyoung (ojyoung@ucsc.edu)
 * @date:  July 3, 2025
 *
 * This file contains the CalibratedContactDetector class, which is used to detect
 * contact points on a 9DTact sensor.
 *
 * https://github.com/linchangyi1/9DTact?tab=readme-ov-file#installation
 *
 * The class loads calibration data from a directory and uses it to rectify images.
 *
"""

def select_image_data(image_path):
    """Select images from processed video to use as reference and sample images.""" 
    """TODO: display image selection window"""
    
    # Load video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
   
    # Select images
    images = []
    for i in range(frame_count):
        ret, frame = video.read()
        

def SAM2_segmentation(image):
    """Use SAM2 segmentation model to segment image."""
    """TODO: load SAM2 segmentation model"""
    
    """TODO: segment image"""
    
    """TODO: return segmented image"""
    
    """TODO: display segmented image"""
    
    """TODO: save segmented image"""
    
    """TODO: return segmented image"""
    
    

def Detectron2_identification(image):
    """TODO: Experiment with Detect"""



class CalibratedContactDetector:
    def __init__(self, sensor_id=1):
        """
        Initialize the calibrated contact detector.
        
        Args:
            sensor_id (int): Sensor ID (default: 1)
        """
        self.sensor_id = sensor_id
        self.calibration_dir = f"calibration/sensor_{sensor_id}/camera_calibration"
        
        # Load calibration data
        self.load_calibration_data()
        self.load_video_calibration_data("calibration/WIN_20250703_15_55_38_Pro.mp4")
        
    def load_calibration_data(self):
        """Load camera calibration data."""
        try:
            # Load distortion correction maps
            self.row_index = np.load(f"{self.calibration_dir}/row_index.npy")
            self.col_index = np.load(f"{self.calibration_dir}/col_index.npy")
            
            # Load position and scale data
            position_scale = np.load(f"{self.calibration_dir}/position_scale.npy")
            self.center_position = position_scale[0:2]
            self.pixel_per_mm = position_scale[2]
            
            print(f"Calibration data loaded successfully")
            print(f"Center position: {self.center_position}")
            print(f"Pixel per mm: {self.pixel_per_mm}")
            
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            raise
    
    def load_video_calibration_data(self, video_path):
        """Load video calibration data."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Could not open video file")
        print(f"Video file opened successfully")
        
        print(f"\nVideo Properties:")
        print(f"Video Frame width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"Video Frame height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"Video Frame count: {self.cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        print(f"Video Frame rate: {self.cap.get(cv2.CAP_PROP_FPS)}")
        
        while True:
            ret, frame = self.cap.read()

            
    
    def rectify_image(self, img):
        """Apply distortion correction to an image."""
        if len(img.shape) == 3:
            # Color image
            img_rectify = img[self.row_index, self.col_index]
        else:
            # Grayscale image
            img_rectify = img[self.row_index, self.col_index]
        return img_rectify
    
    def load_and_rectify_images(self, ref_path, sample_path):
        """Load and rectify reference and sample images."""
        # Load images
        self.ref_img = cv2.imread(ref_path)
        self.sample_img = cv2.imread(sample_path)
        
        if self.ref_img is None or self.sample_img is None:
            raise ValueError("Could not load one or both images")
        
        # Apply distortion correction
        self.ref_rectified = self.rectify_image(self.ref_img)
        self.sample_rectified = self.rectify_image(self.sample_img)
        
        # Convert to grayscale
        self.ref_gray = cv2.cvtColor(self.ref_rectified, cv2.COLOR_BGR2GRAY)
        self.sample_gray = cv2.cvtColor(self.sample_rectified, cv2.COLOR_BGR2GRAY)
        
        print(f"Reference image shape: {self.ref_img.shape} -> {self.ref_rectified.shape}")
        print(f"Sample image shape: {self.sample_img.shape} -> {self.sample_rectified.shape}")
        
    def detect_contact_points(self, threshold=30, min_area=100, blur_kernel=5):
        """
        Detect contact points by comparing calibrated reference and sample images.
        
        Args:
            threshold (int): Intensity difference threshold for contact detection
            min_area (int): Minimum area for contact regions
            blur_kernel (int): Kernel size for Gaussian blur
            
        Returns:
            tuple: (contact_mask, contact_contours, contact_centers)
        """
        # Apply Gaussian blur to reduce noise
        ref_blur = cv2.GaussianBlur(self.ref_gray, (blur_kernel, blur_kernel), 0)
        sample_blur = cv2.GaussianBlur(self.sample_gray, (blur_kernel, blur_kernel), 0)
        
        # Calculate absolute difference
        diff = cv2.absdiff(ref_blur, sample_blur)
        
        # Apply threshold to identify significant changes (contact areas)
        # Since contact makes the membrane more opaque (whiter), we look for areas
        # where the sample is brighter than the reference
        contact_mask = np.zeros_like(diff)
        contact_mask[sample_blur > (ref_blur + threshold)] = 255
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        contact_mask = cv2.morphologyEx(contact_mask, cv2.MORPH_CLOSE, kernel)
        contact_mask = cv2.morphologyEx(contact_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of contact regions
        contours, _ = cv2.findContours(contact_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        filtered_contours = []
        contact_centers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                filtered_contours.append(contour)
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    contact_centers.append((cx, cy))
        
        return contact_mask, filtered_contours, contact_centers
    
    def visualize_results(self, contact_mask, contours, centers, save_path=None):
        """
        Visualize the contact detection results with calibrated images.
        
        Args:
            contact_mask: Binary mask of detected contact areas
            contours: List of contact contours
            centers: List of contact center points
            save_path (str): Optional path to save the visualization
        """
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Original images
        axes[0, 0].imshow(cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Reference Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(self.sample_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Original Sample Image')
        axes[0, 1].axis('off')
        
        # Calibrated images
        axes[0, 2].imshow(cv2.cvtColor(self.ref_rectified, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Calibrated Reference Image')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(cv2.cvtColor(self.sample_rectified, cv2.COLOR_BGR2RGB))
        axes[0, 3].set_title('Calibrated Sample Image')
        axes[0, 3].axis('off')
        
        # Grayscale and analysis
        axes[1, 0].imshow(self.ref_gray, cmap='gray')
        axes[1, 0].set_title('Reference (Grayscale)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(self.sample_gray, cmap='gray')
        axes[1, 1].set_title('Sample (Grayscale)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(contact_mask, cmap='gray')
        axes[1, 2].set_title('Contact Mask')
        axes[1, 2].axis('off')
        
        # Overlay contact points on calibrated sample image
        result_img = self.sample_rectified.copy()
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
        
        # Draw center points
        for center in centers:
            cv2.circle(result_img, center, 5, (255, 0, 0), -1)
            cv2.circle(result_img, center, 10, (255, 0, 0), 2)
        
        axes[1, 3].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[1, 3].set_title('Detected Contact Points')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
    def analyze_contact_intensity(self, contact_mask):
        """
        Analyze the intensity changes in contact areas.
        
        Args:
            contact_mask: Binary mask of detected contact areas
            
        Returns:
            dict: Statistics about contact intensity
        """
        # Get contact regions
        contact_regions = contact_mask > 0
        
        if not np.any(contact_regions):
            return {"error": "No contact regions detected"}
        
        # Calculate intensity statistics
        ref_contact_intensities = self.ref_gray[contact_regions]
        sample_contact_intensities = self.sample_gray[contact_regions]
        
        # Since contact makes the membrane whiter, we calculate the increase in intensity
        intensity_increase = sample_contact_intensities - ref_contact_intensities
        
        stats = {
            "total_contact_pixels": np.sum(contact_regions),
            "mean_intensity_increase": np.mean(intensity_increase),
            "max_intensity_increase": np.max(intensity_increase),
            "std_intensity_increase": np.std(intensity_increase),
            "contact_area_percentage": (np.sum(contact_regions) / contact_regions.size) * 100
        }
        
        return stats

def main():
    """Main function to demonstrate calibrated contact detection."""
    # Paths to your images (relative to shape_reconstruction directory)
    ref_path = "calibration/sensor_1/camera_calibration/ref.png"
    sample_path = "calibration/sensor_1/camera_calibration/sample.png"
    video_path = "calibration/WIN_20250703_15_55_38_Pro.mp4"
    """Change paths to new reference images"""  
    
    """    
    try:
        # Load and rectify images

        # Detect contact points

        
        # Visualize results
        detector.visualize_results(
            contact_mask, contours, centers,
            save_path="calibrated_contact_detection_results.png"
        )
        
    except Exception as e:
        print(f"Error: {e}")
    """
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    image = Image.open('calibration/sensor_1/depth_calibration/sample.png')
    image = np.array(image.convert("RGB"))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()    
    
    sam2_checkpoint = "../../../sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        
        
        
        
if __name__ == "__main__":
    main() 