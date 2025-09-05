
"""
 *
 * @file    contact_detection.py
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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ContactDetector:
    def __init__(self, ref_path, sample_path):
        """
        Initialize the contact detector with reference and sample images.
        
        Args:
            ref_path (str): Path to the reference image (before contact)
            sample_path (str): Path to the sample image (after contact)
        """
        self.ref_path = ref_path
        self.sample_path = sample_path
        self.ref_img = None
        self.sample_img = None
        
    def load_images(self):
        """Load and preprocess the reference and sample images."""
        # Load images
        self.ref_img = cv2.imread(self.ref_path)
        self.sample_img = cv2.imread(self.sample_path)
        
        if self.ref_img is None or self.sample_img is None:
            raise ValueError("Could not load one or both images")
            
        # Convert to grayscale for better contact detection
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        self.sample_gray = cv2.cvtColor(self.sample_img, cv2.COLOR_BGR2GRAY)
        
        print(f"Reference image shape: {self.ref_img.shape}")
        print(f"Sample image shape: {self.sample_img.shape}")
        
    def detect_contact_points(self, threshold=30, min_area=100, blur_kernel=5):
        """
        Detect contact points by comparing reference and sample images.
        
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
        Visualize the contact detection results.
        
        Args:
            contact_mask: Binary mask of detected contact areas
            contours: List of contact contours
            centers: List of contact center points
            save_path (str): Optional path to save the visualization
        """
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original images
        axes[0, 0].imshow(cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Reference Image (Before Contact)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(self.sample_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Sample Image (After Contact)')
        axes[0, 1].axis('off')
        
        # Grayscale images
        axes[0, 2].imshow(self.ref_gray, cmap='gray')
        axes[0, 2].set_title('Reference (Grayscale)')
        axes[0, 2].axis('off')
        
        # Difference and contact detection
        axes[1, 0].imshow(self.sample_gray, cmap='gray')
        axes[1, 0].set_title('Sample (Grayscale)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(contact_mask, cmap='gray')
        axes[1, 1].set_title('Contact Mask')
        axes[1, 1].axis('off')
        
        # Overlay contact points on sample image
        result_img = self.sample_img.copy()
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
        
        # Draw center points
        for center in centers:
            cv2.circle(result_img, center, 5, (255, 0, 0), -1)
            cv2.circle(result_img, center, 10, (255, 0, 0), 2)
        
        axes[1, 2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Detected Contact Points')
        axes[1, 2].axis('off')
        
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
    """Main function to demonstrate contact detection."""
    # Paths to your images (relative to shape_reconstruction directory)
    ref_path = "calibration/sensor_1/depth_calibration/ref.png"
    sample_path = "calibration/sensor_1/depth_calibration/sample.png"
    
    # Initialize detector
    detector = ContactDetector(ref_path, sample_path)
    
    try:
        # Load images
        detector.load_images()
        
        # Detect contact points
        contact_mask, contours, centers = detector.detect_contact_points(
            threshold=25,  # Adjust based on your sensor characteristics
            min_area=50,   # Minimum contact area
            blur_kernel=3  # Blur kernel size
        )
        
        print(f"Detected {len(contours)} contact regions")
        print(f"Contact centers: {centers}")
        
        # Analyze contact intensity
        stats = detector.analyze_contact_intensity(contact_mask)
        print("\nContact Analysis:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Visualize results
        detector.visualize_results(
            contact_mask, contours, centers,
            save_path="contact_detection_results.png"
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 