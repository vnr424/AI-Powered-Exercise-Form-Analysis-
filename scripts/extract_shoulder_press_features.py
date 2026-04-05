import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from pathlib import Path

class FeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def extract_features_from_image(self, image_path):
        """Extract 18 features from image"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        # Extract key landmarks
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        left_shoulder = [landmarks[LEFT_SHOULDER].x, landmarks[LEFT_SHOULDER].y, landmarks[LEFT_SHOULDER].z]
        right_shoulder = [landmarks[RIGHT_SHOULDER].x, landmarks[RIGHT_SHOULDER].y, landmarks[RIGHT_SHOULDER].z]
        left_elbow = [landmarks[LEFT_ELBOW].x, landmarks[LEFT_ELBOW].y, landmarks[LEFT_ELBOW].z]
        right_elbow = [landmarks[RIGHT_ELBOW].x, landmarks[RIGHT_ELBOW].y, landmarks[RIGHT_ELBOW].z]
        left_wrist = [landmarks[LEFT_WRIST].x, landmarks[LEFT_WRIST].y, landmarks[LEFT_WRIST].z]
        right_wrist = [landmarks[RIGHT_WRIST].x, landmarks[RIGHT_WRIST].y, landmarks[RIGHT_WRIST].z]
        left_hip = [landmarks[LEFT_HIP].x, landmarks[LEFT_HIP].y, landmarks[LEFT_HIP].z]
        right_hip = [landmarks[RIGHT_HIP].x, landmarks[RIGHT_HIP].y, landmarks[RIGHT_HIP].z]
        
        # Calculate angles
        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        elbow_angle_diff = abs(left_elbow_angle - right_elbow_angle)
        elbow_symmetry = 100 * (1 - elbow_angle_diff / avg_elbow_angle) if avg_elbow_angle > 0 else 0
        
        # Shoulder features
        shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        
        # Hip features
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        
        # Wrist features
        avg_wrist_x = (left_wrist[0] + right_wrist[0]) / 2
        avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
        avg_wrist_z = (left_wrist[2] + right_wrist[2]) / 2
        wrist_x_normalized = (avg_wrist_x - shoulder_mid_x) / shoulder_width if shoulder_width > 0 else 0
        wrist_y_position = avg_wrist_y - shoulder_mid_x
        
        # Retraction
        baseline_shoulder_width = shoulder_width
        retraction_offset = shoulder_width - baseline_shoulder_width
        retraction_normalized = retraction_offset / baseline_shoulder_width if baseline_shoulder_width > 0 else 0
        
        # Alignment
        alignment_score = (
            0.4 * elbow_symmetry + 
            0.3 * (1 - abs(wrist_x_normalized)) * 100 + 
            0.3 * (1 - abs(retraction_normalized)) * 100
        )
        
        # Binary indicators
        has_flare = 1 if avg_elbow_angle > 75 else 0
        extreme_flare = 1 if avg_elbow_angle > 120 else 0
        
        # Non-linear
        avg_elbow_angle_squared = avg_elbow_angle ** 2
        
        # Return features
        features = {
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            'avg_elbow_angle': avg_elbow_angle,
            'elbow_angle_diff': elbow_angle_diff,
            'elbow_symmetry': elbow_symmetry,
            'wrist_x': avg_wrist_x,
            'wrist_y': avg_wrist_y,
            'wrist_z': avg_wrist_z,
            'wrist_x_normalized': wrist_x_normalized,
            'wrist_y_position': wrist_y_position,
            'shoulder_width': shoulder_width,
            'shoulder_mid_x': shoulder_mid_x,
            'retraction_offset': retraction_offset,
            'retraction_normalized': retraction_normalized,
            'alignment_score': alignment_score,
            'has_flare': has_flare,
            'extreme_flare': extreme_flare,
            'avg_elbow_angle_squared': avg_elbow_angle_squared
        }
        
        return features
    
    def process_folder_recursive(self, folder_path, label):
        """Process all images in folder and subfolders"""
        data = []
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f" Folder not found: {folder_path}")
            return data
        
        # Find all image files recursively
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.rglob(ext))  # rglob = recursive glob
        
        print(f"\n Found {len(image_files)} images in '{folder.name}' (including subfolders)")
        
        success_count = 0
        for idx, img_path in enumerate(image_files, 1):
            # Show subfolder name
            subfolder = img_path.parent.name
            print(f"  [{idx}/{len(image_files)}] {subfolder}/{img_path.name}...", end=' ')
            
            features = self.extract_features_from_image(img_path)
            
            if features:
                features['label'] = label
                features['image_path'] = str(img_path)
                data.append(features)
                success_count += 1
            else:
        
        print(f" Successfully processed {success_count}/{len(image_files)} images")
        return data

if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    print("="*60)
    print("SHOULDER PRESS FEATURE EXTRACTION")
    print("="*60)
    
    # Use Path to handle spaces in folder names
    desktop = Path.home() / "Desktop"
    correct_folder = desktop / "correct shoulder press"
    incorrect_folder = desktop / "incorrect shoulder press"
    
    print(f"\n Correct folder: {correct_folder}")
    print(f"   Subfolders: correct 1, correct 2")
    print(f" Incorrect folder: {incorrect_folder}")
    print(f"   Subfolders: wide, close")
    
    # Process correct images (will search in correct 1 and correct 2)
    print("\n" + "="*60)
    print("PROCESSING CORRECT FORM IMAGES")
    print("="*60)
    correct_data = extractor.process_folder_recursive(correct_folder, label=1)
    
    # Process incorrect images (will search in wide and close)
    print("\n" + "="*60)
    print("PROCESSING INCORRECT FORM IMAGES")
    print("="*60)
    incorrect_data = extractor.process_folder_recursive(incorrect_folder, label=0)
    
    # Combine data
    all_data = correct_data + incorrect_data
    
    if len(all_data) == 0:
        print("\n No data extracted!")
        print("\nMake sure images exist in:")
        print(f"  {correct_folder}/correct 1/")
        print(f"  {correct_folder}/correct 2/")
        print(f"  {incorrect_folder}/wide/")
        print(f"  {incorrect_folder}/close/")
    else:
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Create output directory
        output_dir = Path('../data')
        output_dir.mkdir(exist_ok=True)
        
        # Save to CSV
        output_file = output_dir / 'shoulder_press_features.csv'
        df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print(" EXTRACTION COMPLETE!")
        print("="*60)
        print(f"\n Total samples: {len(df)}")
        print(f"    Correct form: {len(df[df['label']==1])}")
        print(f"    Incorrect form: {len(df[df['label']==0])}")
        print(f"\n Saved to: {output_file}")
        print("\n Ready for retraining!")
