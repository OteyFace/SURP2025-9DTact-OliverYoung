"""Calibration Method Comparison and Force Estimation

This script compares different contact detection methods:
1. SAM-2 segmentation (AI-based)
2. OpenCV traditional methods (morphological operations)
3. Original 9DTact approach

It also provides force estimation based on pixel color values from contact regions.
"""

import cv2
import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

from New_Camera import Camera
from New_Contact_Region_Detection import segment_image_sam2, detect_contact_region

@dataclass
class ContactAnalysis:
    """Data structure for contact region analysis results."""
    area: float  # Contact area in pixels
    intensity_mean: float  # Mean intensity of contact region
    intensity_std: float  # Standard deviation of intensity
    transparency: float  # Estimated transparency (0-1)
    depth_map: np.ndarray  # Estimated depth map
    force_estimate: float  # Estimated force value
    method: str  # Detection method used

class CalibrationMethodComparison:
    """Compare different contact detection and calibration methods."""
    
    def __init__(self, cfg_path: str = 'shape_config.yaml', sensor_id: int = 3):
        """Initialize the comparison framework."""
        self.cfg = self._load_config(cfg_path, sensor_id)
        self.camera = Camera(self.cfg, calibrated=True, file_mode=True)
        
    def _load_config(self, cfg_path: str, sensor_id: int) -> dict:
        """Load configuration file."""
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg['sensor_id'] = sensor_id
        return cfg
    
    def sam2_detection(self, img_path: str) -> ContactAnalysis:
        """SAM-2 based contact detection."""
        # Load and preprocess image
        bgr = self._load_and_preprocess(img_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Get SAM-2 segments
        segs = segment_image_sam2(bgr)
        
        # Create contact mask from SAM-2 segments
        mask = np.zeros_like(gray, dtype=np.uint8)
        if segs:
            for ann in segs:
                seg_mask = ann["segmentation"]
                if seg_mask.shape != mask.shape:
                    seg_mask = cv2.resize(seg_mask.astype(np.uint8), 
                                        (mask.shape[1], mask.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                mask[seg_mask] = 255
        
        return self._analyze_contact(bgr, mask, "SAM-2")
    
    def opencv_detection(self, img_path: str) -> ContactAnalysis:
        """OpenCV traditional contact detection."""
        # Load and preprocess image
        bgr = self._load_and_preprocess(img_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Traditional high-pass filtering approach
        # 1. Background estimation
        bg = cv2.GaussianBlur(gray, (51, 51), 0)
        
        # 2. High-frequency detail extraction
        diff = gray.astype(np.int16) - bg.astype(np.int16)
        mag = np.abs(diff).astype(np.uint8)
        
        # 3. Thresholding (lower threshold for higher sensitivity)
        _, mask = cv2.threshold(mag, 5, 255, cv2.THRESH_BINARY)
        
        # 4. Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # smaller kernel preserves fine contacts
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return self._analyze_contact(bgr, mask, "OpenCV")
    
    def original_9dtact_detection(self, img_path: str, ref_img_path: str) -> ContactAnalysis:
        """Original 9DTact approach using reference image."""
        # Load images
        bgr = self._load_and_preprocess(img_path)
        ref_bgr = self._load_and_preprocess(ref_img_path)
        
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        
        # Difference-based detection
        diff = cv2.absdiff(gray, ref_gray)
        
        # Threshold the difference
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return self._analyze_contact(bgr, mask, "Original 9DTact")
    
    def _load_and_preprocess(self, img_path: str) -> np.ndarray:
        """Load and preprocess image using camera calibration."""
        self.camera.file_path = img_path
        return self.camera.get_rectify_crop_image()
    
    def _analyze_contact(self, bgr: np.ndarray, mask: np.ndarray, method: str) -> ContactAnalysis:
        """Analyze contact region and estimate properties."""
        # Calculate contact area
        area = np.sum(mask > 0)
        
        # Analyze intensity in contact region
        contact_pixels = bgr[mask > 0]
        if len(contact_pixels) > 0:
            # Convert to grayscale for intensity analysis
            contact_gray = cv2.cvtColor(contact_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY)
            intensity_mean = np.mean(contact_gray)
            intensity_std = np.std(contact_gray)
        else:
            intensity_mean = 0
            intensity_std = 0
        
        # Estimate transparency (based on intensity variation)
        transparency = 1.0 - (intensity_std / 255.0) if intensity_std > 0 else 0.0
        
        # Create depth map (simplified - you can enhance this)
        depth_map = self._create_depth_map(bgr, mask, intensity_mean)
        
        # Estimate force based on area and intensity
        force_estimate = self._estimate_force(area, intensity_mean, transparency)
        
        return ContactAnalysis(
            area=area,
            intensity_mean=intensity_mean,
            intensity_std=intensity_std,
            transparency=transparency,
            depth_map=depth_map,
            force_estimate=force_estimate,
            method=method
        )
    
    def _create_depth_map(self, bgr: np.ndarray, mask: np.ndarray, intensity_mean: float) -> np.ndarray:
        """Create a depth map based on contact analysis."""
        # Convert to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Create depth map where contact regions have depth based on intensity
        depth_map = np.zeros_like(gray, dtype=np.float32)
        
        # Normalize intensity to depth (0-1 range)
        normalized_intensity = gray.astype(np.float32) / 255.0
        
        # Apply mask and create depth map
        depth_map[mask > 0] = normalized_intensity[mask > 0]
        
        # Smooth the depth map
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
        return depth_map
    
    def _estimate_force(self, area: float, intensity_mean: float, transparency: float) -> float:
        """Estimate force based on contact properties."""
        # Simple force estimation model
        # You can calibrate these parameters with your force sensor data
        
        # Force is proportional to area and intensity
        area_factor = area / 1000.0  # Normalize area
        intensity_factor = intensity_mean / 255.0  # Normalize intensity
        transparency_factor = transparency
        
        # Combined force estimate
        force = area_factor * intensity_factor * (1 + transparency_factor)
        
        return force
    
    def compare_methods(self, img_path: str, ref_img_path: Optional[str] = None) -> Dict[str, ContactAnalysis]:
        """Compare all detection methods on the same image."""
        results = {}
        
        # SAM-2 detection
        try:
            results['sam2'] = self.sam2_detection(img_path)
        except Exception as e:
            print(f"SAM-2 detection failed: {e}")
        
        # OpenCV detection
        try:
            results['opencv'] = self.opencv_detection(img_path)
        except Exception as e:
            print(f"OpenCV detection failed: {e}")
        
        # Original 9DTact detection (requires reference image)
        if ref_img_path and os.path.exists(ref_img_path):
            try:
                results['original_9dtact'] = self.original_9dtact_detection(img_path, ref_img_path)
            except Exception as e:
                print(f"Original 9DTact detection failed: {e}")
        
        return results
    
    def create_force_vector_field(self, contact_analysis: ContactAnalysis) -> Tuple[np.ndarray, np.ndarray]:
        """Create force vector field based on depth map gradients."""
        depth_map = contact_analysis.depth_map
        
        # Calculate gradients
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Normalize gradients to force magnitude
        force_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        force_magnitude = force_magnitude * contact_analysis.force_estimate
        
        return grad_x, grad_y, force_magnitude
    
    def save_comparison_results(self, results: Dict[str, ContactAnalysis], output_dir: str):
        """Save comparison results and visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison table
        comparison_data = []
        for method, analysis in results.items():
            comparison_data.append({
                'Method': analysis.method,
                'Area (pixels)': analysis.area,
                'Mean Intensity': analysis.intensity_mean,
                'Intensity Std': analysis.intensity_std,
                'Transparency': analysis.transparency,
                'Force Estimate': analysis.force_estimate
            })
        
        # Save comparison table
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        df.to_csv(os.path.join(output_dir, 'method_comparison.csv'), index=False)
        
        # Create visualizations
        self._create_comparison_plots(results, output_dir)
    
    def _create_comparison_plots(self, results: Dict[str, ContactAnalysis], output_dir: str):
        """Create comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Area comparison
        methods = list(results.keys())
        areas = [results[m].area for m in methods]
        axes[0, 0].bar(methods, areas)
        axes[0, 0].set_title('Contact Area Comparison')
        axes[0, 0].set_ylabel('Area (pixels)')
        
        # Intensity comparison
        intensities = [results[m].intensity_mean for m in methods]
        axes[0, 1].bar(methods, intensities)
        axes[0, 1].set_title('Mean Intensity Comparison')
        axes[0, 1].set_ylabel('Intensity')
        
        # Force estimate comparison
        forces = [results[m].force_estimate for m in methods]
        axes[1, 0].bar(methods, forces)
        axes[1, 0].set_title('Force Estimate Comparison')
        axes[1, 0].set_ylabel('Force Estimate')
        
        # Transparency comparison
        transparencies = [results[m].transparency for m in methods]
        axes[1, 1].bar(methods, transparencies)
        axes[1, 1].set_title('Transparency Comparison')
        axes[1, 1].set_ylabel('Transparency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to demonstrate the comparison framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare contact detection methods')
    parser.add_argument('--img', type=str, required=True, help='Path to test image')
    parser.add_argument('--ref', type=str, help='Path to reference image (for 9DTact method)')
    parser.add_argument('--output', type=str, default='calibration/comparison_results', 
                       help='Output directory for results')
    parser.add_argument('--cfg', type=str, default='shape_config.yaml', help='Config file')
    parser.add_argument('--sensor_id', type=int, default=3, help='Sensor ID')
    
    args = parser.parse_args()
    
    # Initialize comparison framework
    comparator = CalibrationMethodComparison(args.cfg, args.sensor_id)
    
    # Run comparison
    results = comparator.compare_methods(args.img, args.ref)
    
    # Save results
    comparator.save_comparison_results(results, args.output)
    
    # Print summary
    print("\n=== Contact Detection Method Comparison ===")
    for method, analysis in results.items():
        print(f"\n{analysis.method}:")
        print(f"  Area: {analysis.area:.0f} pixels")
        print(f"  Mean Intensity: {analysis.intensity_mean:.1f}")
        print(f"  Force Estimate: {analysis.force_estimate:.3f}")
        print(f"  Transparency: {analysis.transparency:.3f}")
    
    print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
