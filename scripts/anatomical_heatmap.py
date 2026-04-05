import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class AnatomicalHeatmap:
    """
    Generate Grad-CAM style heatmaps for exercise form analysis
    Uses Random Forest feature importance to highlight problem areas
    """
    
    def __init__(self, model):
        """Initialize with trained model"""
        self.model = model
        
        # Feature to body part mapping
        self.feature_bodypart_map = {
            # Elbow features
            'left_elbow_angle': ['left_elbow'],
            'right_elbow_angle': ['right_elbow'],
            'avg_elbow_angle': ['left_elbow', 'right_elbow'],
            'elbow_angle_diff': ['left_elbow', 'right_elbow'],
            'elbow_symmetry': ['left_elbow', 'right_elbow'],
            'has_flare': ['left_elbow', 'right_elbow'],
            'extreme_flare': ['left_elbow', 'right_elbow'],
            'avg_elbow_angle_squared': ['left_elbow', 'right_elbow'],
            
            # Wrist features
            'wrist_x': ['left_wrist', 'right_wrist'],
            'wrist_y': ['left_wrist', 'right_wrist'],
            'wrist_z': ['left_wrist', 'right_wrist'],
            'wrist_x_normalized': ['left_wrist', 'right_wrist'],
            'wrist_y_position': ['left_wrist', 'right_wrist'],
            
            # Shoulder features
            'shoulder_width': ['left_shoulder', 'right_shoulder'],
            'shoulder_mid_x': ['left_shoulder', 'right_shoulder'],
            
            # Torso features
            'retraction_offset': ['torso'],
            'retraction_normalized': ['torso'],
            'alignment_score': ['torso'],
        }
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                'left_elbow_angle', 'right_elbow_angle', 'avg_elbow_angle',
                'wrist_x', 'wrist_y', 'wrist_z',
                'shoulder_width', 'shoulder_mid_x', 'retraction_offset',
                'wrist_y_position', 'elbow_angle_diff', 'elbow_symmetry',
                'wrist_x_normalized', 'retraction_normalized', 'alignment_score',
                'has_flare', 'extreme_flare', 'avg_elbow_angle_squared'
            ]
            self.feature_importance = dict(zip(feature_names, model.feature_importances_))
        else:
            self.feature_importance = {}
    
    def calculate_bodypart_intensity(self, features, prediction, errors):
        """
        Calculate heatmap intensity for each body part
        Combines feature importance + feature deviation + error severity
        
        Returns: dict mapping joint names to intensity (0-1)
        """
        intensities = {
            'left_shoulder': 0.0,
            'right_shoulder': 0.0,
            'left_elbow': 0.0,
            'right_elbow': 0.0,
            'left_wrist': 0.0,
            'right_wrist': 0.0,
            'torso': 0.0,
        }
        
        # Calculate intensity from feature deviations
        for feature_name, bodyparts in self.feature_bodypart_map.items():
            if feature_name not in features:
                continue
            
            value = features[feature_name]
            importance = self.feature_importance.get(feature_name, 0.1)
            deviation = self._calculate_deviation(feature_name, value)
            
            # Contribution = importance * deviation
            contribution = importance * deviation
            
            for bodypart in bodyparts:
                intensities[bodypart] += contribution
        
        # Boost intensity for body parts mentioned in errors
        if errors:
            error_boost = {
                'left_elbow': 0.0,
                'right_elbow': 0.0,
                'left_shoulder': 0.0,
                'right_shoulder': 0.0,
            }
            
            for error in errors:
                error_type = error.error_type.lower()
                severity_multiplier = 1.5 if error.severity == "CRITICAL" else 1.0
                
                if 'elbow' in error_type:
                    error_boost['left_elbow'] += 0.3 * severity_multiplier
                    error_boost['right_elbow'] += 0.3 * severity_multiplier
                if 'shoulder' in error_type:
                    error_boost['left_shoulder'] += 0.2 * severity_multiplier
                    error_boost['right_shoulder'] += 0.2 * severity_multiplier
            
            for part, boost in error_boost.items():
                intensities[part] += boost
        
        # Normalize to 0-1 range
        max_intensity = max(intensities.values())
        if max_intensity > 0:
            intensities = {k: min(v / max_intensity, 1.0) for k, v in intensities.items()}
        
        return intensities
    
    def _calculate_deviation(self, feature_name, value):
        """How much does this feature deviate from correct range (0-1)"""
        
        # Elbow angle features
        if 'elbow_angle' in feature_name and 'squared' not in feature_name and 'diff' not in feature_name:
            # Optimal: 45-75 degrees
            if 45 <= value <= 75:
                return 0.0
            elif value > 120:
                return 1.0  # Extreme flare
            elif value > 75:
                return (value - 75) / 45
            else:
                return (45 - value) / 45
        
        elif feature_name == 'elbow_angle_diff':
            # Should be < 15 degrees
            return min(value / 30, 1.0)
        
        elif feature_name == 'elbow_symmetry':
            # Should be close to 1.0
            return abs(1.0 - value)
        
        elif 'extreme_flare' in feature_name:
            return float(value) * 1.5
        
        elif 'has_flare' in feature_name:
            return float(value)
        
        else:
            return 0.3  # Default moderate deviation
    
    def generate_heatmap_overlay(self, frame, landmarks, prediction, features, errors):
        """
        Generate beautiful Grad-CAM style heatmap overlay
        
        Args:
            frame: Video frame
            landmarks: MediaPipe pose landmarks
            prediction: Model prediction (0=incorrect, 1=correct)
            features: Feature dictionary
            errors: List of detected errors
            
        Returns:
            Frame with heatmap overlay
        """
        h, w = frame.shape[:2]
        
        # Calculate intensities (always generate for Grade F reps)
        intensities = self.calculate_bodypart_intensity(features, prediction, errors)
        
        # Force minimum intensity if all zero (fallback for correct prediction frames)
        if all(v < 0.1 for v in intensities.values()):
            intensities = {k: 0.4 for k in intensities}
            if errors:
                for e in errors:
                    if 'elbow' in e.error_type.lower():
                        intensities['left_elbow'] = 0.9
                        intensities['right_elbow'] = 0.9
                    if 'shoulder' in e.error_type.lower():
                        intensities['left_shoulder'] = 0.8
                        intensities['right_shoulder'] = 0.8
        
        # Create heatmap layer
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Landmark indices
        LANDMARKS = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
        }
        
        lm = landmarks[0]
        
        # Draw gaussian blobs for each body part
        for part_name, intensity in intensities.items():
            if intensity < 0.1:
                continue  # Skip low-intensity parts
            
            if part_name == 'torso':
                # Draw between shoulders and hips
                left_shoulder = lm[11]
                right_shoulder = lm[12]
                center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
                center_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
                radius = 80
            else:
                # Get landmark position
                landmark_idx = LANDMARKS.get(part_name)
                if landmark_idx is None:
                    continue
                
                point = lm[landmark_idx]
                center_x = int(point.x * w)
                center_y = int(point.y * h)
                
                # Radius based on intensity
                radius = int(60 + 40 * intensity)
            
            # Draw gaussian blob
            self._add_gaussian_blob(heatmap, center_x, center_y, radius, intensity)
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply color map (jet: blue=cold/safe, red=hot/problem)
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create mask for blending (only where heatmap > 0)
        mask = (heatmap > 0.05).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = np.expand_dims(mask, axis=2)
        
        # Blend heatmap with original frame
        alpha = 0.65  # Stronger overlay for visibility
        result = frame.astype(np.float32) * (1 - alpha * mask) + heatmap_colored.astype(np.float32) * alpha * mask
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Add legend
        result = self._add_legend(result, intensities)
        
        return result
    
    def _add_gaussian_blob(self, heatmap, cx, cy, radius, intensity):
        """Add a gaussian blob to the heatmap"""
        h, w = heatmap.shape
        
        # Create meshgrid
        y, x = np.ogrid[0:h, 0:w]
        
        # Calculate distance from center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Gaussian function
        sigma = radius / 2.5
        gaussian = intensity * np.exp(-(dist ** 2) / (2 * sigma ** 2))
        
        # Add to heatmap
        heatmap += gaussian
    
    def _add_legend(self, frame, intensities):
        """Add color legend showing intensity scale"""
        h, w = frame.shape[:2]
        
        # Legend position (top right)
        legend_w = 30
        legend_h = 150
        legend_x = w - 60
        legend_y = 20
        
        # Create gradient bar
        gradient = np.linspace(0, 1, legend_h)
        gradient = np.tile(gradient[:, np.newaxis], (1, legend_w))
        gradient_colored = cv2.applyColorMap((gradient * 255).astype(np.uint8), cv2.COLORMAP_JET)
        gradient_colored = cv2.flip(gradient_colored, 0)  # Flip so red is on top
        
        # Add border
        cv2.rectangle(gradient_colored, (0, 0), (legend_w-1, legend_h-1), (255, 255, 255), 2)
        
        # Place on frame
        frame[legend_y:legend_y+legend_h, legend_x:legend_x+legend_w] = gradient_colored
        
        # Add labels
        cv2.putText(frame, "HIGH", (legend_x - 45, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "RISK", (legend_x - 45, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, "LOW", (legend_x - 40, legend_y + legend_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def generate_standalone_heatmap(self, frame, landmarks, prediction, features, errors):
        """
        Generate standalone heatmap image (no original frame)
        Similar to the middle column in medical Grad-CAM images
        
        Returns:
            Heatmap image (BGR format for OpenCV)
        """
        h, w = frame.shape[:2]
        intensities = self.calculate_bodypart_intensity(features, prediction, errors)
        
        # Force minimum intensities based on errors if all zero
        if all(v < 0.1 for v in intensities.values()):
            intensities = {k: 0.3 for k in intensities}
            for e in errors:
                if 'elbow' in e.error_type.lower():
                    intensities['left_elbow'] = 0.9
                    intensities['right_elbow'] = 0.9
                if 'shoulder' in e.error_type.lower():
                    intensities['left_shoulder'] = 0.8
                    intensities['right_shoulder'] = 0.8
        
        # Create black background
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        LANDMARKS = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
        }
        
        lm = landmarks[0]
        
        for part_name, intensity in intensities.items():
            if intensity < 0.1:
                continue
            
            if part_name == 'torso':
                left_shoulder = lm[11]
                right_shoulder = lm[12]
                center_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
                center_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
                radius = 80
            else:
                landmark_idx = LANDMARKS.get(part_name)
                if landmark_idx is None:
                    continue
                point = lm[landmark_idx]
                center_x = int(point.x * w)
                center_y = int(point.y * h)
                radius = int(60 + 40 * intensity)
            
            self._add_gaussian_blob(heatmap, center_x, center_y, radius, intensity)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap_colored
    
    def generate_report_visualization(self, frame, landmarks, prediction, features, errors):
        """
        Generate 3-panel visualization for medical report
        Similar to: Original | Heatmap | Overlay
        
        Returns:
            Combined 3-panel image
        """
        
        h, w = frame.shape[:2]
        
        # Panel 1: Original frame with skeleton
        panel1 = frame.copy()
        
        # Panel 2: Standalone heatmap
        panel2 = self.generate_standalone_heatmap(frame, landmarks, prediction, features, errors)
        if panel2 is None:
            panel2 = np.zeros_like(frame)
        
        # Panel 3: Overlay
        panel3 = self.generate_heatmap_overlay(frame, landmarks, prediction, features, errors)
        
        # Combine panels horizontally
        # Add labels
        label_height = 40
        combined = np.zeros((h + label_height, w * 3, 3), dtype=np.uint8)
        
        # Add labels
        cv2.putText(combined, "ORIGINAL", (w//2 - 50, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "HEATMAP (Problem Areas)", (w + w//2 - 110, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "OVERLAY", (2*w + w//2 - 50, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Place panels
        combined[label_height:, 0:w] = panel1
        combined[label_height:, w:2*w] = panel2
        combined[label_height:, 2*w:3*w] = panel3
        
        return combined

def integrate_heatmap_into_system(frame, landmarks, prediction, features, errors, model, heatmap_enabled=True):
    """
    Easy integration function for existing system
    
    Args:
        frame: Video frame
        landmarks: MediaPipe landmarks
        prediction: Model prediction
        features: Feature dict
        errors: Error list
        model: Trained model
        heatmap_enabled: Whether to show heatmap
        
    Returns:
        Frame with or without heatmap overlay
    """
    if not heatmap_enabled or prediction == 1:
        return frame
    
    heatmap_gen = AnatomicalHeatmap(model)
    return heatmap_gen.generate_heatmap_overlay(frame, landmarks, prediction, features, errors)