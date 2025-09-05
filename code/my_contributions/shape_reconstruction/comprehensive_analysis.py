"""Comprehensive Tactile Sensing Analysis

This script provides a complete pipeline for:
1. Contact detection using multiple methods (SAM-2, OpenCV, 9DTact)
2. Force estimation from pixel properties
3. 3D force vector field generation
4. Depth map creation and analysis
"""

import cv2
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass

from New_Camera import Camera
from New_Contact_Region_Detection import isolate_contact_area
from calibration_method_comparison import CalibrationMethodComparison, ContactAnalysis
from force_calibration import ForceCalibrator, ForceVectorEstimator

@dataclass
class ComprehensiveAnalysis:
    """Complete analysis results for a tactile sensing image."""
    image_path: str
    contact_analyses: Dict[str, ContactAnalysis]
    force_vectors: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    depth_maps: Dict[str, np.ndarray]
    isolated_contacts: Dict[str, np.ndarray]
    best_method: str
    total_force_estimate: float

class TactileSensingAnalyzer:
    """Complete tactile sensing analysis pipeline."""
    
    def __init__(self, cfg_path: str = 'shape_config.yaml', sensor_id: int = 3):
        """Initialize the analyzer."""
        self.cfg = self._load_config(cfg_path, sensor_id)
        self.camera = Camera(self.cfg, calibrated=True, file_mode=True)
        self.comparator = CalibrationMethodComparison(cfg_path, sensor_id)
        self.force_calibrator = None  # Will be set if calibration is available
        
    def _load_config(self, cfg_path: str, sensor_id: int) -> dict:
        """Load configuration file."""
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg['sensor_id'] = sensor_id
        return cfg
    
    def load_force_calibration(self, calibration_path: str) -> None:
        """Load force calibration data."""
        if os.path.exists(calibration_path):
            self.force_calibrator = ForceCalibrator()
            self.force_calibrator.load_calibration(calibration_path)
            print("Force calibration loaded successfully")
        else:
            print("No force calibration found. Using basic force estimation.")
    
    def analyze_image(self, img_path: str, ref_img_path: Optional[str] = None) -> ComprehensiveAnalysis:
        """Perform comprehensive analysis of a tactile sensing image."""
        print(f"Analyzing image: {img_path}")
        
        # 1. Contact detection using multiple methods
        contact_analyses = self.comparator.compare_methods(img_path, ref_img_path)
        
        # 2. Create isolated contact images
        isolated_contacts = {}
        depth_maps = {}
        force_vectors = {}
        
        for method, analysis in contact_analyses.items():
            # Load and preprocess image
            self.camera.file_path = img_path
            bgr = self.camera.get_rectify_crop_image()
            
            # Create contact mask
            mask = np.zeros_like(analysis.depth_map, dtype=np.uint8)
            mask[analysis.depth_map > 0] = 255
            
            # Isolate contact area
            isolated_contacts[method] = isolate_contact_area(bgr, mask)
            
            # Store depth map
            depth_maps[method] = analysis.depth_map
            
            # Create force vectors if calibrator is available
            if self.force_calibrator:
                vector_estimator = ForceVectorEstimator(self.force_calibrator)
                force_vectors[method] = vector_estimator.estimate_force_vectors(
                    analysis, analysis.depth_map
                )
        
        # 3. Determine best method (based on contact area and consistency)
        best_method = self._select_best_method(contact_analyses)
        
        # 4. Calculate total force estimate
        total_force = 0.0
        if best_method in contact_analyses:
            total_force = contact_analyses[best_method].force_estimate
        
        return ComprehensiveAnalysis(
            image_path=img_path,
            contact_analyses=contact_analyses,
            force_vectors=force_vectors,
            depth_maps=depth_maps,
            isolated_contacts=isolated_contacts,
            best_method=best_method,
            total_force_estimate=total_force
        )
    
    def _select_best_method(self, contact_analyses: Dict[str, ContactAnalysis]) -> str:
        """Select the best contact detection method."""
        if not contact_analyses:
            return "none"
        
        # Score each method based on:
        # 1. Contact area (larger is better, but not too large)
        # 2. Intensity consistency (lower std is better)
        # 3. Method reliability
        
        scores = {}
        for method, analysis in contact_analyses.items():
            score = 0
            
            # Area score (prefer medium-sized contacts)
            area_score = min(analysis.area / 1000.0, 1.0)  # Normalize
            score += area_score * 0.4
            
            # Intensity consistency score
            consistency_score = 1.0 - (analysis.intensity_std / 255.0)
            score += consistency_score * 0.3
            
            # Method preference score
            if method == 'sam2':
                score += 0.3  # Prefer SAM-2
            elif method == 'opencv':
                score += 0.2
            elif method == 'original_9dtact':
                score += 0.1
            
            scores[method] = score
        
        return max(scores, key=scores.get)
    
    def create_comprehensive_visualization(self, analysis: ComprehensiveAnalysis, 
                                        output_dir: str) -> None:
        """Create comprehensive visualization of analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create multi-panel visualization (show OpenCV row only if available)
        display_methods = [(m, ca) for m, ca in analysis.contact_analyses.items() if m.lower() == 'opencv']
        if not display_methods:
            display_methods = list(analysis.contact_analyses.items())
        n_methods = len(display_methods)
        if n_methods == 0:
            return
        
        # Load original image
        self.camera.file_path = analysis.image_path
        bgr = self.camera.get_rectify_crop_image()
        
        # Create visualization
        fig, axes = plt.subplots(n_methods, 4, figsize=(16, 4 * n_methods))
        rows = axes if n_methods > 1 else [axes]
        
        for i, (method, contact_analysis) in enumerate(display_methods):
            row_array = rows[i] if isinstance(rows, np.ndarray) else rows[i]
            # Flatten to a simple list of 4 axes
            row_list = list(np.ravel(row_array))
            ax0, ax1, ax2, ax3 = row_list[:4]
            
            # Original image
            ax0.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ax0.set_title(f'{method.upper()} - Original')
            ax0.axis('off')
            
            # Isolated contact
            if method in analysis.isolated_contacts:
                isolated = analysis.isolated_contacts[method]
                ax1.imshow(cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB))
                ax1.set_title(f'{method.upper()} - Isolated Contact')
                ax1.axis('off')
            
            # Depth map
            if method in analysis.depth_maps:
                depth_map = analysis.depth_maps[method]
                im = ax2.imshow(depth_map, cmap='viridis', vmin=0.0, vmax=1.5)
                ax2.set_title(f'{method.upper()} - Depth Map')
                ax2.axis('off')
                cbar = plt.colorbar(im, ax=ax2)
                cbar.set_label('Height (mm)')
            
            # Force vectors (if available)
            if method in analysis.force_vectors:
                grad_x, grad_y, force_mag = analysis.force_vectors[method]
                
                # Sample points for vector visualization
                h, w = force_mag.shape
                step = max(1, min(h, w) // 15)
                y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
                u = grad_x[y_coords, x_coords]
                v = grad_y[y_coords, x_coords]
                
                ax3.quiver(x_coords, y_coords, u, v, force_mag[y_coords, x_coords],
                           cmap='hot', scale=50)
                ax3.set_title(f'{method.upper()} - Force Vectors')
                ax3.axis('off')
            else:
                continue
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_analysis_results(self, analysis: ComprehensiveAnalysis, output_dir: str) -> None:
        """Save analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual images
        for method, isolated_contact in analysis.isolated_contacts.items():
            cv2.imwrite(os.path.join(output_dir, f'{method}_isolated_contact.png'), 
                       isolated_contact)
        
        for method, depth_map in analysis.depth_maps.items():
            # Normalize depth map for saving
            depth_normalized = ((depth_map - depth_map.min()) / 
                              (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f'{method}_depth_map.png'), 
                       depth_normalized)
        
        # Save analysis summary
        summary_data = []
        for method, contact_analysis in analysis.contact_analyses.items():
            summary_data.append({
                'Method': method,
                'Area': contact_analysis.area,
                'Mean_Intensity': contact_analysis.intensity_mean,
                'Intensity_Std': contact_analysis.intensity_std,
                'Transparency': contact_analysis.transparency,
                'Force_Estimate': contact_analysis.force_estimate,
                'Is_Best_Method': method == analysis.best_method
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'analysis_summary.csv'), index=False)
        
        # Save force vectors as numpy arrays
        for method, (grad_x, grad_y, force_mag) in analysis.force_vectors.items():
            np.savez(os.path.join(output_dir, f'{method}_force_vectors.npz'),
                    grad_x=grad_x, grad_y=grad_y, force_magnitude=force_mag)
    
    def batch_analyze(self, image_paths: List[str], ref_img_path: Optional[str] = None,
                     output_dir: str = 'calibration/comprehensive_analysis') -> List[ComprehensiveAnalysis]:
        """Analyze multiple images in batch."""
        results = []
        
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
            
            try:
                analysis = self.analyze_image(img_path, ref_img_path)
                results.append(analysis)
                
                # Save individual results
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                img_output_dir = os.path.join(output_dir, img_name)
                self.save_analysis_results(analysis, img_output_dir)
                self.create_comprehensive_visualization(analysis, img_output_dir)
                
            except Exception as e:
                print(f"Error analyzing {img_path}: {e}")
                continue
        
        # Create batch summary
        self._create_batch_summary(results, output_dir)
        
        return results
    
    def _create_batch_summary(self, analyses: List[ComprehensiveAnalysis], output_dir: str) -> None:
        """Create summary of batch analysis."""
        if not analyses:
            return
        
        # Collect summary statistics
        summary_data = []
        for analysis in analyses:
            img_name = os.path.basename(analysis.image_path)
            best_analysis = analysis.contact_analyses.get(analysis.best_method)
            
            if best_analysis:
                summary_data.append({
                    'Image': img_name,
                    'Best_Method': analysis.best_method,
                    'Total_Force': analysis.total_force_estimate,
                    'Contact_Area': best_analysis.area,
                    'Mean_Intensity': best_analysis.intensity_mean,
                    'Transparency': best_analysis.transparency
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, 'batch_summary.csv'), index=False)
        
        # Create batch visualization
        self._create_batch_visualization(analyses, output_dir)
    
    def _create_batch_visualization(self, analyses: List[ComprehensiveAnalysis], output_dir: str) -> None:
        """Create batch visualization."""
        if not analyses:
            return
        
        # Extract data for plotting
        methods = list(analyses[0].contact_analyses.keys())
        method_data = {method: [] for method in methods}
        
        for analysis in analyses:
            for method in methods:
                if method in analysis.contact_analyses:
                    method_data[method].append(analysis.contact_analyses[method].area)
                else:
                    method_data[method].append(0)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Contact area comparison
        x = range(len(analyses))
        for method in methods:
            axes[0, 0].plot(x, method_data[method], marker='o', label=method.upper())
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Contact Area (pixels)')
        axes[0, 0].set_title('Contact Area Comparison')
        axes[0, 0].legend()
        
        # Force estimates
        forces = [analysis.total_force_estimate for analysis in analyses]
        axes[0, 1].plot(x, forces, marker='s', color='red')
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Force Estimate')
        axes[0, 1].set_title('Force Estimates')
        
        # Method selection frequency
        method_counts = {}
        for analysis in analyses:
            method = analysis.best_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        methods_list = list(method_counts.keys())
        counts = list(method_counts.values())
        axes[1, 0].bar(methods_list, counts)
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Best Method Selection')
        
        # Intensity distribution
        all_intensities = []
        for analysis in analyses:
            for contact_analysis in analysis.contact_analyses.values():
                all_intensities.append(contact_analysis.intensity_mean)
        
        axes[1, 1].hist(all_intensities, bins=20, alpha=0.7)
        axes[1, 1].set_xlabel('Mean Intensity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Intensity Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function for comprehensive analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive tactile sensing analysis')
    parser.add_argument('--img', type=str, required=True, help='Path to test image')
    parser.add_argument('--ref', type=str, help='Path to reference image')
    parser.add_argument('--calibration', type=str, help='Path to force calibration file')
    parser.add_argument('--output', type=str, default='calibration/comprehensive_analysis',
                       help='Output directory')
    parser.add_argument('--batch', type=str, help='Directory with multiple images for batch analysis')
    parser.add_argument('--cfg', type=str, default='shape_config.yaml', help='Config file')
    parser.add_argument('--sensor_id', type=int, default=3, help='Sensor ID')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TactileSensingAnalyzer(args.cfg, args.sensor_id)
    
    # Load force calibration if available
    if args.calibration:
        analyzer.load_force_calibration(args.calibration)
    
    if args.batch:
        # Batch analysis
        import glob
        image_paths = glob.glob(os.path.join(args.batch, '*.jpg'))
        image_paths.extend(glob.glob(os.path.join(args.batch, '*.png')))
        
        if image_paths:
            print(f"Found {len(image_paths)} images for batch analysis")
            results = analyzer.batch_analyze(image_paths, args.ref, args.output)
            print(f"Batch analysis complete. Results saved to: {args.output}")
        else:
            print(f"No images found in {args.batch}")
    else:
        # Single image analysis
        analysis = analyzer.analyze_image(args.img, args.ref)
        
        # Save results
        analyzer.save_analysis_results(analysis, args.output)
        analyzer.create_comprehensive_visualization(analysis, args.output)
        
        # Print summary
        print(f"\n=== Comprehensive Analysis Results ===")
        print(f"Best method: {analysis.best_method}")
        print(f"Total force estimate: {analysis.total_force_estimate:.3f}")
        print(f"Results saved to: {args.output}")

if __name__ == '__main__':
    main()
