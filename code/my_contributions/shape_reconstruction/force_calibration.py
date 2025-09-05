"""Force Calibration and Estimation Module

This module provides tools for calibrating force estimation from tactile sensor images
using external force sensor data. It creates a mapping between pixel properties and
actual force measurements.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import json
import os

class ForceCalibrator:
    """Calibrate force estimation using external force sensor data."""
    
    def __init__(self):
        """Initialize the force calibrator."""
        self.model = None
        self.calibration_data = []
        self.feature_names = ['area', 'intensity_mean', 'intensity_std', 'transparency']
        
    def add_calibration_point(self, image_path: str, force_measurement: float, 
                             contact_analysis: 'ContactAnalysis') -> None:
        """Add a calibration data point."""
        features = [
            contact_analysis.area,
            contact_analysis.intensity_mean,
            contact_analysis.intensity_std,
            contact_analysis.transparency
        ]
        
        self.calibration_data.append({
            'image_path': image_path,
            'features': features,
            'measured_force': force_measurement,
            'estimated_force': contact_analysis.force_estimate
        })
    
    def train_model(self, model_type: str = 'linear') -> None:
        """Train a force estimation model."""
        if len(self.calibration_data) < 5:
            raise ValueError("Need at least 5 calibration points to train model")
        
        # Prepare training data
        X = np.array([point['features'] for point in self.calibration_data])
        y = np.array([point['measured_force'] for point in self.calibration_data])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Save model coefficients
        if hasattr(self.model, 'coef_'):
            coefficients = dict(zip(self.feature_names, self.model.coef_))
            coefficients['intercept'] = self.model.intercept_
            self.model_coefficients = coefficients
        else:
            self.model_coefficients = None
    
    def estimate_force(self, contact_analysis: 'ContactAnalysis') -> float:
        """Estimate force using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        features = np.array([
            contact_analysis.area,
            contact_analysis.intensity_mean,
            contact_analysis.intensity_std,
            contact_analysis.transparency
        ]).reshape(1, -1)
        
        return self.model.predict(features)[0]
    
    def save_calibration(self, output_path: str) -> None:
        """Save calibration data and model."""
        calibration_data = {
            'calibration_points': self.calibration_data,
            'model_coefficients': self.model_coefficients,
            'feature_names': self.feature_names
        }
        
        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2, default=str)
    
    def load_calibration(self, input_path: str) -> None:
        """Load calibration data and model."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.calibration_data = data['calibration_points']
        self.model_coefficients = data['model_coefficients']
        self.feature_names = data['feature_names']
    
    def create_calibration_report(self, output_dir: str) -> None:
        """Create a comprehensive calibration report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create calibration plots
        self._create_calibration_plots(output_dir)
        
        # Create calibration table
        self._create_calibration_table(output_dir)
        
        # Create model performance report
        if self.model is not None:
            self._create_model_report(output_dir)
    
    def _create_calibration_plots(self, output_dir: str) -> None:
        """Create calibration plots."""
        if not self.calibration_data:
            return
        
        # Extract data
        measured_forces = [point['measured_force'] for point in self.calibration_data]
        estimated_forces = [point['estimated_force'] for point in self.calibration_data]
        areas = [point['features'][0] for point in self.calibration_data]
        intensities = [point['features'][1] for point in self.calibration_data]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Measured vs Estimated Force
        axes[0, 0].scatter(measured_forces, estimated_forces, alpha=0.7)
        axes[0, 0].plot([min(measured_forces), max(measured_forces)], 
                        [min(measured_forces), max(measured_forces)], 'r--')
        axes[0, 0].set_xlabel('Measured Force')
        axes[0, 0].set_ylabel('Estimated Force')
        axes[0, 0].set_title('Measured vs Estimated Force')
        
        # Area vs Force
        axes[0, 1].scatter(areas, measured_forces, alpha=0.7)
        axes[0, 1].set_xlabel('Contact Area (pixels)')
        axes[0, 1].set_ylabel('Measured Force')
        axes[0, 1].set_title('Contact Area vs Force')
        
        # Intensity vs Force
        axes[1, 0].scatter(intensities, measured_forces, alpha=0.7)
        axes[1, 0].set_xlabel('Mean Intensity')
        axes[1, 0].set_ylabel('Measured Force')
        axes[1, 0].set_title('Intensity vs Force')
        
        # Force distribution
        axes[1, 1].hist(measured_forces, bins=10, alpha=0.7)
        axes[1, 1].set_xlabel('Force')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Force Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calibration_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_calibration_table(self, output_dir: str) -> None:
        """Create calibration data table."""
        df_data = []
        for i, point in enumerate(self.calibration_data):
            df_data.append({
                'Point': i + 1,
                'Image': os.path.basename(point['image_path']),
                'Area': point['features'][0],
                'Intensity_Mean': point['features'][1],
                'Intensity_Std': point['features'][2],
                'Transparency': point['features'][3],
                'Measured_Force': point['measured_force'],
                'Estimated_Force': point['estimated_force'],
                'Error': abs(point['measured_force'] - point['estimated_force'])
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(os.path.join(output_dir, 'calibration_data.csv'), index=False)
    
    def _create_model_report(self, output_dir: str) -> None:
        """Create model performance report."""
        if self.model_coefficients:
            report = {
                'model_type': type(self.model).__name__,
                'coefficients': self.model_coefficients,
                'feature_importance': dict(zip(self.feature_names, self.model.coef_))
            }
            
            with open(os.path.join(output_dir, 'model_report.json'), 'w') as f:
                json.dump(report, f, indent=2)

class ForceVectorEstimator:
    """Estimate force vectors from contact regions."""
    
    def __init__(self, force_calibrator: ForceCalibrator):
        """Initialize with a trained force calibrator."""
        self.force_calibrator = force_calibrator
    
    def estimate_force_vectors(self, contact_analysis: 'ContactAnalysis', 
                             depth_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate force vectors from contact analysis and depth map."""
        # Calculate depth gradients
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Estimate total force
        total_force = self.force_calibrator.estimate_force(contact_analysis)
        
        # Normalize gradients to force magnitude
        force_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Scale by total force
        if np.max(force_magnitude) > 0:
            force_magnitude = force_magnitude * (total_force / np.max(force_magnitude))
        
        return grad_x, grad_y, force_magnitude
    
    def create_force_visualization(self, bgr: np.ndarray, contact_analysis: 'ContactAnalysis',
                                 depth_map: np.ndarray, output_path: str) -> None:
        """Create force vector visualization."""
        grad_x, grad_y, force_magnitude = self.estimate_force_vectors(contact_analysis, depth_map)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Depth map
        im1 = axes[0, 1].imshow(depth_map, cmap='viridis')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Force magnitude
        im2 = axes[1, 0].imshow(force_magnitude, cmap='hot')
        axes[1, 0].set_title('Force Magnitude')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Force vectors (quiver plot)
        # Sample points for vector visualization
        h, w = depth_map.shape
        step = max(1, min(h, w) // 20)  # Sample every 20th point
        
        y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
        u = grad_x[y_coords, x_coords]
        v = grad_y[y_coords, x_coords]
        
        axes[1, 1].quiver(x_coords, y_coords, u, v, force_magnitude[y_coords, x_coords],
                          cmap='hot', scale=50)
        axes[1, 1].set_title('Force Vectors')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Example usage of force calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Force calibration and estimation')
    parser.add_argument('--mode', choices=['calibrate', 'estimate'], required=True,
                       help='Calibration or estimation mode')
    parser.add_argument('--data', type=str, help='Path to calibration data CSV')
    parser.add_argument('--output', type=str, default='calibration/force_calibration',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='linear',
                       choices=['linear', 'ridge', 'random_forest'],
                       help='Model type for force estimation')
    
    args = parser.parse_args()
    
    calibrator = ForceCalibrator()
    
    if args.mode == 'calibrate':
        # Load calibration data
        if args.data and os.path.exists(args.data):
            df = pd.read_csv(args.data)
            
            # Add calibration points (you'll need to implement this based on your data format)
            for _, row in df.iterrows():
                # This is a placeholder - you'll need to adapt this to your data format
                print(f"Processing calibration point: {row}")
                
            # Train model
            calibrator.train_model(args.model)
            
            # Save calibration
            os.makedirs(args.output, exist_ok=True)
            calibrator.save_calibration(os.path.join(args.output, 'force_calibration.json'))
            calibrator.create_calibration_report(args.output)
            
            print(f"Calibration saved to: {args.output}")
    
    elif args.mode == 'estimate':
        # Load calibration
        calibration_path = os.path.join(args.output, 'force_calibration.json')
        if os.path.exists(calibration_path):
            calibrator.load_calibration(calibration_path)
            print("Calibration loaded successfully")
        else:
            print("No calibration found. Please run calibration first.")

if __name__ == '__main__':
    main()
