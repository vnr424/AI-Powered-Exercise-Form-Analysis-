import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from enhanced_explainability import AnatomicalExpertSystem, draw_anatomical_overlay
from explainability_system import FeatureVisualizer
from rep_counter import RepCounter, RepAnalysis
from audio_coach import AudioCoach
from exercise_config import ExerciseConfig
from anatomical_heatmap import AnatomicalHeatmap
from person_detection_filter import PersonIsolator, integrate_person_filter

MODEL_PATH = "models/random_forest_tuned.joblib"
SCALER_PATH = "models/feature_scaler_tuned.joblib"
POSE_MODEL_PATH = "models/pose_landmarker_heavy.task"

# System configuration
SYSTEM_STATE = {
    'heatmap_enabled': True,
    'person_filter_enabled': True
}

print("Loading models...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

if not Path(POSE_MODEL_PATH).exists():
    print("ERROR: Pose model not found!")
    exit(1)

print("Models loaded successfully")

# Initialize systems
anatomical_expert = AnatomicalExpertSystem()
visualizer = FeatureVisualizer(model)
heatmap_generator = AnatomicalHeatmap(model)
print("Medical expert system and heatmap visualization initialized")

LANDMARKS = {
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24
}

class ReportViewer:
    """Interactive report viewer with scrolling and heatmap support"""
    
    def __init__(self):
        self.active = False
        self.report_text = ""
        self.scroll_position = 0
        self.max_scroll = 0
        self.report_title = "Medical Report"
        self.heatmap_image = None
        self.current_frame = None
        self.current_landmarks = None
        self.current_prediction = None
        self.current_features = None
        self.current_errors = None
        self.rep_heatmaps = []
        self.rep_heatmap_lookup = {}

    def show(self, report_text, title="Medical Analysis Report", 
             frame=None, landmarks=None, prediction=None, features=None, errors=None,
             rep_heatmaps=None):
        self.active = True
        self.report_text = report_text
        self.report_title = title
        self.scroll_position = 0
        lines = report_text.split('\n')
        self.max_scroll = max(0, len(lines) - 30)
        
        self.current_frame = frame
        self.current_landmarks = landmarks
        self.current_prediction = prediction
        self.current_features = features
        self.current_errors = errors
        
        if rep_heatmaps:
            self.rep_heatmaps = rep_heatmaps
            # Build lookup: rep_number -> heatmap image
            self.rep_heatmap_lookup = {hm['rep_number']: hm for hm in rep_heatmaps}
        else:
            self.rep_heatmaps = []
            self.rep_heatmap_lookup = {}
        
        if frame is not None and landmarks is not None and prediction == 0:
            self.generate_heatmap_visualization()

    def generate_heatmap_visualization(self):
        """Generate 3-panel heatmap visualization for report"""
        if self.current_frame is None or self.current_landmarks is None:
            return
        
        try:
            self.heatmap_image = heatmap_generator.generate_report_visualization(
                frame=self.current_frame.copy(),
                landmarks=self.current_landmarks,
                prediction=self.current_prediction,
                features=self.current_features,
                errors=self.current_errors
            )
        except Exception as e:
            print(f"Warning: Could not generate heatmap: {e}")
            self.heatmap_image = None

    def save_heatmap(self, filename=None):
        """Save heatmap visualization to file"""
        if self.heatmap_image is None:
            return False
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"heatmap_report_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, self.heatmap_image)
            return True
        except Exception as e:
            print(f"Error saving heatmap: {e}")
            return False
    
    def save_all_rep_heatmaps(self, timestamp):
        """Save all rep heatmaps from the set"""
        saved_files = []
        for i, heatmap_data in enumerate(self.rep_heatmaps):
            filename = f"rep_{heatmap_data['rep_number']}_grade_{heatmap_data['grade']}_heatmap_{timestamp}.jpg"
            try:
                cv2.imwrite(filename, heatmap_data['image'])
                saved_files.append(filename)
            except Exception as e:
                print(f"Error saving rep {i+1} heatmap: {e}")
        return saved_files

    def hide(self):
        self.active = False
        self.combined_heatmap = None

    def scroll_up(self):
        self.scroll_position = max(0, self.scroll_position - 3)

    def scroll_down(self):
        self.scroll_position = min(self.max_scroll, self.scroll_position + 3)
    
    def build_combined_heatmap(self):
        """Build combined heatmap from rep heatmaps"""
        if not self.rep_heatmaps:
            return None
        try:
            heatmaps = [hm['image'] for hm in self.rep_heatmaps]
            if len(heatmaps) == 1:
                return heatmaps[0]
            elif len(heatmaps) <= 4:
                return np.vstack(heatmaps)
            else:
                mid = (len(heatmaps) + 1) // 2
                row1 = np.hstack(heatmaps[:mid])
                row2 = np.hstack(heatmaps[mid:])
                if row1.shape[1] != row2.shape[1]:
                    diff = abs(row1.shape[1] - row2.shape[1])
                    pad = np.zeros((row2.shape[0] if row1.shape[1] > row2.shape[1] else row1.shape[0], diff, 3), dtype=np.uint8)
                    if row1.shape[1] > row2.shape[1]:
                        row2 = np.hstack([row2, pad])
                    else:
                        row1 = np.hstack([row1, pad])
                return np.vstack([row1, row2])
        except Exception as e:
            print(f"Could not build combined heatmap: {e}")
            return None

    def draw(self, frame):
        """Draw report overlay on frame"""
        if not self.active:
            return frame

        h, w = frame.shape[:2]

        # Dark background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (40, 40), (w-40, h-40), (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.15, overlay, 0.85, 0)

        # Border
        cv2.rectangle(frame, (40, 40), (w-40, h-40), (0, 165, 255), 4)

        # Title bar
        cv2.rectangle(frame, (40, 40), (w-40, 90), (0, 165, 255), -1)
        cv2.putText(frame, self.report_title.upper(), (55, 70),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Controls hint
        cv2.putText(frame, "ESC/R: Close | UP/DOWN/J/K: Scroll | S: Save",
                   (w-530, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Content rendering
        lines = self.report_text.split('\n')
        y = 110
        line_height = 20
        visible_area_height = h - 150
        max_visible_lines = visible_area_height // line_height

        end_line = min(len(lines), self.scroll_position + max_visible_lines)
        visible_lines = lines[self.scroll_position:end_line]

        for line in visible_lines:
            if y >= h - 60:
                break

            # Color coding based on content
            if "ERROR #" in line:
                color, weight, size = (50, 200, 255), 2, 0.65
            elif "CRITICAL" in line and "SEVERITY" in line:
                color, weight, size = (0, 0, 255), 2, 0.6
            elif "SEVERITY: HIGH" in line:
                color, weight, size = (0, 100, 255), 2, 0.6
            elif "SEVERITY:" in line:
                color, weight, size = (0, 165, 255), 2, 0.6
            elif any(kw in line for kw in ["AFFECTED ANATOMY:", "INJURY RISK:", "CORRECTION:",
                                            "JOINTS AT RISK:", "AFFECTED MUSCLES:",
                                            "BIOMECHANICAL", "RECOMMENDATIONS"]):
                color, weight, size = (100, 255, 255), 2, 0.58
            elif line.strip().startswith(("-", "-", "*")):
                color, weight, size = (220, 220, 220), 1, 0.52
            elif "CORRECT" in line:
                color, weight, size = (0, 255, 0), 1, 0.52
            elif line.strip().startswith(("X", "WARNING")):
                color, weight, size = (0, 165, 255), 1, 0.52
            elif "===" in line or "---" in line:
                color, weight, size = (120, 120, 120), 1, 0.5
            elif "REP #" in line or "LAST REP" in line:
                color, weight, size = (0, 255, 255), 2, 0.6
            elif any(kw in line for kw in ["Grade:", "GRADE", "Status:", "Duration:"]):
                color, weight, size = (0, 255, 100), 2, 0.58
            else:
                color, weight, size = (255, 255, 255), 1, 0.52

            display_text = line[:135] + "..." if len(line) > 135 else line
            try:
                cv2.putText(frame, display_text, (60, y),
                           cv2.FONT_HERSHEY_SIMPLEX, size, color, weight)
            except:
                pass

            y += line_height

            # Inline heatmap thumbnail next to each REP # line
            if "REP #" in line and hasattr(self, 'rep_heatmap_lookup') and self.rep_heatmap_lookup:
                try:
                    import re as _re
                    m = _re.search(r'REP #(\d+)', line)
                    if m:
                        rep_num = int(m.group(1))
                        if rep_num in self.rep_heatmap_lookup:
                            hm_data  = self.rep_heatmap_lookup[rep_num]
                            hm_img   = hm_data['image']
                            hm_grade = hm_data['grade']
                            hm_h, hm_w = hm_img.shape[:2]
                            thumb_h  = 70
                            scale    = thumb_h / hm_h
                            thumb_w  = int(hm_w * scale)
                            if thumb_w > 0:
                                thumb    = cv2.resize(hm_img, (thumb_w, thumb_h))
                                tx = w - thumb_w - 65
                                ty = y - line_height - 5
                                if tx > 200 and ty >= 100 and ty + thumb_h < h - 50:
                                    grade_colors = {'A':(0,200,0),'B':(180,180,0),'C':(0,165,255),'D':(0,80,255),'F':(0,0,220)}
                                    gc = grade_colors.get(hm_grade, (180,180,180))
                                    cv2.rectangle(frame, (tx-2, ty-14), (tx+thumb_w+2, ty+thumb_h+2), gc, 1)
                                    cv2.putText(frame, f"Grade {hm_grade}", (tx, ty-3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gc, 1)
                                    frame[ty:ty+thumb_h, tx:tx+thumb_w] = thumb
                except Exception:
                    pass

        # Show combined heatmap when scrolled to bottom
        at_bottom = self.scroll_position >= self.max_scroll or (self.max_scroll <= 3)
        if at_bottom and self.rep_heatmaps:
            combined = self.build_combined_heatmap()
            if combined is not None:
                # Fixed heatmap area: use bottom 45% of the report box
                hm_start_y = int(h * 0.55)
                hm_end_y   = h - 50
                hm_start_x = 50
                hm_end_x   = w - 50
                hm_area_h  = hm_end_y - hm_start_y - 30
                hm_area_w  = hm_end_x - hm_start_x

                # Dark background for heatmap area
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (hm_start_x, hm_start_y), (hm_end_x, hm_end_y), (10,10,10), -1)
                frame = cv2.addWeighted(frame, 0.2, overlay2, 0.8, 0)

                # Header bar
                cv2.rectangle(frame, (hm_start_x, hm_start_y), (hm_end_x, hm_start_y+24), (0,100,180), -1)
                cv2.putText(frame, f"GRAD-CAM HEATMAPS ({len(self.rep_heatmaps)} reps)  |  Blue=Safe  |  Red=Critical Injury Risk",
                           (hm_start_x+5, hm_start_y+17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                # Resize and draw heatmap
                hm_h, hm_w = combined.shape[:2]
                scale  = min(hm_area_w / hm_w, hm_area_h / hm_h)
                new_w  = max(1, int(hm_w * scale))
                new_h  = max(1, int(hm_h * scale))
                resized = cv2.resize(combined, (new_w, new_h))
                x_off  = hm_start_x + (hm_area_w - new_w) // 2
                y_off  = hm_start_y + 26
                frame[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        # Scrollbar
        if len(lines) > max_visible_lines:
            scrollbar_x = w - 60
            scrollbar_top = 100
            scrollbar_height = h - 150

            cv2.rectangle(frame, (scrollbar_x, scrollbar_top),
                         (scrollbar_x + 10, scrollbar_top + scrollbar_height),
                         (80, 80, 80), -1)

            thumb_height = max(30, int(scrollbar_height * (max_visible_lines / len(lines))))
            scroll_range = len(lines) - max_visible_lines
            scroll_percent = self.scroll_position / scroll_range if scroll_range > 0 else 0
            thumb_y = int(scrollbar_top + scroll_percent * (scrollbar_height - thumb_height))

            cv2.rectangle(frame, (scrollbar_x, thumb_y),
                         (scrollbar_x + 10, thumb_y + thumb_height),
                         (0, 165, 255), -1)

            cv2.putText(frame, f"Line {self.scroll_position + 1} / {len(lines)}",
                       (w - 150, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Hint at bottom when heatmaps available
        if self.rep_heatmaps and not at_bottom:
            cv2.putText(frame, f"Scroll to bottom to view {len(self.rep_heatmaps)} heatmap(s)",
                       (60, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        return frame

def select_exercise():
    """Exercise selection interface"""
    print("\nExercise Selection")
    print("-" * 50)

    exercises = ExerciseConfig.get_available_exercises()

    print("\nAvailable exercises:")
    for i, (key, name) in enumerate(exercises, 1):
        config = ExerciseConfig.get_exercise_config(key)
        print(f"  {i}. {name}")
        print(f"     {config['description']}")
        print(f"     Camera: {config['camera_position']}")

    while True:
        try:
            choice = input(f"\nSelect exercise (1-{len(exercises)}): ").strip()
            choice_idx = int(choice) - 1

            if 0 <= choice_idx < len(exercises):
                selected_key, selected_name = exercises[choice_idx]
                config = ExerciseConfig.get_exercise_config(selected_key)

                print(f"\nSelected: {selected_name}")
                print(f"Required landmarks: {', '.join(config['required_landmarks'])}")
                print(f"Optimal elbow angle: {config['elbow_angle_range'][0]}-{config['elbow_angle_range'][1]} degrees")

                return selected_key, config
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            exit(0)

def generate_current_analysis_report(features, prediction, confidence, last_rep, exercise_name):
    """Generate current technique analysis report"""
    report = []
    report.append("-" * 70)
    report.append(f"CURRENT TECHNIQUE ANALYSIS - {exercise_name.upper()}")
    report.append("-" * 70)
    report.append("")

    if prediction == 1:
        report.append("CLASSIFICATION: CORRECT TECHNIQUE")
        report.append(f"  Model Confidence: {confidence*100:.1f}%")
    else:
        report.append("CLASSIFICATION: INCORRECT TECHNIQUE")
        report.append(f"  Model Confidence: {confidence*100:.1f}%")

    report.append("")
    report.append("BIOMECHANICAL MEASUREMENTS")
    report.append("-" * 70)
    report.append("")

    left_elbow = features.get('left_elbow_angle', 0)
    right_elbow = features.get('right_elbow_angle', 0)
    avg_elbow = features.get('avg_elbow_angle', 0)

    report.append("ELBOW ANGLES:")
    report.append(f"  Left Elbow:  {left_elbow:.1f} degrees")
    report.append(f"  Right Elbow: {right_elbow:.1f} degrees")
    report.append(f"  Average:     {avg_elbow:.1f} degrees")
    report.append(f"  Difference:  {abs(left_elbow - right_elbow):.1f} degrees")
    report.append("")

    if avg_elbow < 45:
        report.append("  Status: Too tucked - consider increasing elbow angle")
    elif avg_elbow <= 75:
        report.append("  Status: Optimal range (45-75 degrees)")
    elif avg_elbow <= 120:
        report.append("  Status: Flared - reduce elbow angle")
    else:
        report.append("  Status: CRITICAL - Extreme elbow flare")

    report.append("")

    symmetry_score = features.get('symmetry_score', 0)
    symmetry_grade = features.get('overall_symmetry_grade', 'C')

    report.append("SYMMETRY ANALYSIS")
    report.append("-" * 70)
    report.append("")
    report.append(f"Overall Symmetry Grade: {symmetry_grade}")
    report.append(f"Symmetry Score: {symmetry_score:.3f} (lower is better)")
    report.append("")

    shoulder_diff = features.get('shoulder_height_diff', 0) * 100
    elbow_diff = features.get('elbow_height_diff', 0) * 100
    wrist_diff = features.get('wrist_height_diff', 0) * 100

    report.append("Vertical Position Symmetry:")
    report.append(f"  Shoulders: {shoulder_diff:.1f}% {'Good' if shoulder_diff < 5 else 'Asymmetric'}")
    report.append(f"  Elbows:    {elbow_diff:.1f}% {'Good' if elbow_diff < 5 else 'Asymmetric'}")
    report.append(f"  Wrists:    {wrist_diff:.1f}% {'Good' if wrist_diff < 5 else 'Asymmetric'}")

    if features.get('has_hip_data', False):
        hip_diff = features.get('hip_height_diff', 0) * 100
        report.append(f"  Hips:      {hip_diff:.1f}% {'Good' if hip_diff < 5 else 'Asymmetric'}")

    report.append("")

    elbow_angle_diff = features.get('elbow_angle_diff', 0)
    shoulder_angle_diff = features.get('shoulder_angle_diff', 0)

    report.append("Angular Symmetry:")
    report.append(f"  Elbow Angles:    {elbow_angle_diff:.1f} degrees {'Good' if elbow_angle_diff < 15 else 'Asymmetric'}")
    report.append(f"  Shoulder Angles: {shoulder_angle_diff:.1f} degrees {'Good' if shoulder_angle_diff < 15 else 'Asymmetric'}")
    report.append("")

    report.append("BAR PATH ANALYSIS")
    report.append("-" * 70)
    report.append("")

    bar_tilt = features.get('bar_vertical_tilt', 0) * 100
    bar_center = features.get('bar_center_offset', 0) * 100

    report.append(f"Vertical Tilt:  {bar_tilt:.1f}% {'Level' if bar_tilt < 5 else 'Uneven'}")
    if bar_tilt >= 5:
        report.append("  Bar is not level - check hand placement and grip")

    report.append(f"Center Offset:  {bar_center:.1f}% {'Centered' if bar_center < 15 else 'Off-center'}")
    report.append("")

    if last_rep:
        report.append("LAST COMPLETED REP")
        report.append("-" * 70)
        report.append("")
        report.append(f"  Rep Number: #{last_rep.rep_number}")
        report.append(f"  Quality Grade: {last_rep.get_quality_grade()}")
        report.append(f"  Technique: {'CORRECT' if last_rep.prediction == 1 else 'INCORRECT'}")
        report.append(f"  Confidence: {last_rep.confidence*100:.1f}%")
        report.append(f"  Symmetry Grade: {last_rep.symmetry_grade}")
        report.append(f"  Duration: {last_rep.rep_duration:.1f}s")
        report.append(f"  Elbow Range: {last_rep.elbow_angle_min:.1f} - {last_rep.elbow_angle_max:.1f} degrees")
        report.append(f"  Max Bar Tilt: {last_rep.bar_tilt_max*100:.1f}%")
        report.append(f"  Max Elbow Flare: {last_rep.elbow_flare_max:.1f} degrees")
        report.append("")

        if last_rep.errors:
            report.append("  Errors Detected:")
            for error in last_rep.errors:
                report.append(f"    - {error.error_type} - {error.severity}")
        else:
            report.append("  No errors detected")
        report.append("")

    report.append("-" * 70)
    report.append("END OF ANALYSIS")
    report.append("-" * 70)

    return "\n".join(report)

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2))

def extract_features(pose_landmarks, exercise_config):
    """Extract biomechanical features from pose landmarks"""
    lm = pose_landmarks[0]

    left_shoulder  = lm[LANDMARKS['left_shoulder']]
    right_shoulder = lm[LANDMARKS['right_shoulder']]
    left_elbow     = lm[LANDMARKS['left_elbow']]
    right_elbow    = lm[LANDMARKS['right_elbow']]
    left_wrist     = lm[LANDMARKS['left_wrist']]
    right_wrist    = lm[LANDMARKS['right_wrist']]

    use_hip_features = exercise_config['use_hip_features']

    if use_hip_features:
        left_hip  = lm[LANDMARKS['left_hip']]
        right_hip = lm[LANDMARKS['right_hip']]

    # Calculate elbow angles
    left_elbow_angle  = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    avg_elbow_angle   = (left_elbow_angle + right_elbow_angle) / 2

    # Calculate shoulder angles
    if use_hip_features:
        left_shoulder_angle  = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
    else:
        left_shoulder_angle  = calculate_angle(left_wrist, left_shoulder, left_elbow)
        right_shoulder_angle = calculate_angle(right_wrist, right_shoulder, right_elbow)

    # Wrist position (bar path)
    wrist_x = (left_wrist.x + right_wrist.x) / 2
    wrist_y = (left_wrist.y + right_wrist.y) / 2
    wrist_z = (left_wrist.z + right_wrist.z) / 2

    # Bar path metrics
    bar_vertical_tilt   = abs(left_wrist.y - right_wrist.y)
    bar_horizontal_offset = abs(left_wrist.x - right_wrist.x)
    bar_center_x        = (left_wrist.x + right_wrist.x) / 2
    bar_center_offset   = abs(bar_center_x - 0.5)

    # Symmetry measurements
    shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
    elbow_height_diff    = abs(left_elbow.y - right_elbow.y)
    wrist_height_diff    = abs(left_wrist.y - right_wrist.y)

    if use_hip_features:
        hip_height_diff  = abs(left_hip.y - right_hip.y)
        left_hip_y_val   = left_hip.y
        right_hip_y_val  = right_hip.y
    else:
        hip_height_diff  = 0.0
        left_hip_y_val   = 0.0
        right_hip_y_val  = 0.0

    center_x = 0.5
    shoulder_horizontal_symmetry = abs(
        abs(left_shoulder.x - center_x) - abs(right_shoulder.x - center_x)
    )
    elbow_horizontal_symmetry = abs(
        abs(left_elbow.x - center_x) - abs(right_elbow.x - center_x)
    )

    elbow_angle_diff   = abs(left_elbow_angle - right_elbow_angle)
    shoulder_angle_diff = abs(left_shoulder_angle - right_shoulder_angle)

    # Overall symmetry score
    if use_hip_features:
        symmetry_score = (
            shoulder_height_diff * 2.0 +
            elbow_height_diff * 1.5 +
            wrist_height_diff * 1.5 +
            hip_height_diff * 1.0 +
            (elbow_angle_diff / 100.0) +
            (shoulder_angle_diff / 100.0)
        ) / 7.5
    else:
        symmetry_score = (
            shoulder_height_diff * 2.5 +
            elbow_height_diff * 2.0 +
            wrist_height_diff * 2.0 +
            (elbow_angle_diff / 100.0) +
            (shoulder_angle_diff / 100.0)
        ) / 6.5

    has_shoulder_asymmetry = shoulder_height_diff > 0.05 or shoulder_angle_diff > 15
    has_elbow_asymmetry    = elbow_height_diff > 0.05 or elbow_angle_diff > 15
    has_wrist_asymmetry    = wrist_height_diff > 0.05
    has_hip_tilt           = hip_height_diff > 0.05 if use_hip_features else False

    shoulder_width  = calculate_distance(left_shoulder, right_shoulder)
    shoulder_mid_x  = (left_shoulder.x + right_shoulder.x) / 2

    if use_hip_features:
        torso_mid_x       = (left_hip.x + right_hip.x) / 2
        retraction_offset = abs(shoulder_mid_x - torso_mid_x)
    else:
        elbow_mid_x       = (left_elbow.x + right_elbow.x) / 2
        retraction_offset = abs(shoulder_mid_x - elbow_mid_x)

    return {
        'left_elbow_angle':           left_elbow_angle,
        'right_elbow_angle':          right_elbow_angle,
        'avg_elbow_angle':            avg_elbow_angle,
        'wrist_x':                    wrist_x,
        'wrist_y':                    wrist_y,
        'wrist_z':                    wrist_z,
        'shoulder_width':             shoulder_width,
        'shoulder_mid_x':             shoulder_mid_x,
        'retraction_offset':          retraction_offset,
        'wrist_y_position':           wrist_y,
        'elbow_angle_diff':           elbow_angle_diff,
        'elbow_symmetry':             min(left_elbow_angle, right_elbow_angle) / (max(left_elbow_angle, right_elbow_angle) + 1e-6),
        'wrist_x_normalized':         wrist_x / (shoulder_width + 1e-6),
        'retraction_normalized':      retraction_offset / (shoulder_width + 1e-6),
        'alignment_score':            abs(shoulder_mid_x - wrist_x),
        'has_flare':                  1 if avg_elbow_angle > 75 else 0,
        'extreme_flare':              1 if avg_elbow_angle > 120 else 0,
        'avg_elbow_angle_squared':    avg_elbow_angle ** 2,
        'left_wrist_x':               left_wrist.x,
        'left_wrist_y':               left_wrist.y,
        'right_wrist_x':              right_wrist.x,
        'right_wrist_y':              right_wrist.y,
        'bar_vertical_tilt':          bar_vertical_tilt,
        'bar_horizontal_offset':      bar_horizontal_offset,
        'bar_center_offset':          bar_center_offset,
        'left_shoulder_x':            left_shoulder.x,
        'left_shoulder_y':            left_shoulder.y,
        'right_shoulder_x':           right_shoulder.x,
        'right_shoulder_y':           right_shoulder.y,
        'left_elbow_x':               left_elbow.x,
        'left_elbow_y':               left_elbow.y,
        'right_elbow_x':              right_elbow.x,
        'right_elbow_y':              right_elbow.y,
        'left_shoulder_angle':        left_shoulder_angle,
        'right_shoulder_angle':       right_shoulder_angle,
        'left_hip_y':                 left_hip_y_val,
        'right_hip_y':                right_hip_y_val,
        'shoulder_height_diff':       shoulder_height_diff,
        'elbow_height_diff':          elbow_height_diff,
        'wrist_height_diff':          wrist_height_diff,
        'hip_height_diff':            hip_height_diff,
        'shoulder_horizontal_symmetry': shoulder_horizontal_symmetry,
        'elbow_horizontal_symmetry':  elbow_horizontal_symmetry,
        'shoulder_angle_diff':        shoulder_angle_diff,
        'symmetry_score':             symmetry_score,
        'has_shoulder_asymmetry':     1 if has_shoulder_asymmetry else 0,
        'has_elbow_asymmetry':        1 if has_elbow_asymmetry else 0,
        'has_wrist_asymmetry':        1 if has_wrist_asymmetry else 0,
        'has_hip_tilt':               1 if has_hip_tilt else 0,
        'has_any_asymmetry':          1 if (has_shoulder_asymmetry or has_elbow_asymmetry or has_wrist_asymmetry) else 0,
        'overall_symmetry_grade':     'A' if symmetry_score < 0.05 else ('B' if symmetry_score < 0.10 else ('C' if symmetry_score < 0.15 else 'D')),
        'exercise_type':              exercise_config.get('name', 'Unknown'),
        'has_hip_data':               use_hip_features,
    }

def predict_technique(features_dict):
    """Predict exercise technique using trained model"""
    model_features = {
        'left_elbow_angle':        features_dict['left_elbow_angle'],
        'right_elbow_angle':       features_dict['right_elbow_angle'],
        'avg_elbow_angle':         features_dict['avg_elbow_angle'],
        'wrist_x':                 features_dict['wrist_x'],
        'wrist_y':                 features_dict['wrist_y'],
        'wrist_z':                 features_dict['wrist_z'],
        'shoulder_width':          features_dict['shoulder_width'],
        'shoulder_mid_x':          features_dict['shoulder_mid_x'],
        'retraction_offset':       features_dict['retraction_offset'],
        'wrist_y_position':        features_dict['wrist_y_position'],
        'elbow_angle_diff':        features_dict['elbow_angle_diff'],
        'elbow_symmetry':          features_dict['elbow_symmetry'],
        'wrist_x_normalized':      features_dict['wrist_x_normalized'],
        'retraction_normalized':   features_dict['retraction_normalized'],
        'alignment_score':         features_dict['alignment_score'],
        'has_flare':               features_dict['has_flare'],
        'extreme_flare':           features_dict['extreme_flare'],
        'avg_elbow_angle_squared': features_dict['avg_elbow_angle_squared'],
    }
    features_df     = pd.DataFrame([model_features])
    features_scaled = scaler.transform(features_df)
    prediction      = model.predict(features_scaled)[0]
    probability     = model.predict_proba(features_scaled)[0]
    return prediction, probability[prediction]

# Drawing functions for visualization
def draw_skeleton(image, landmarks, features=None):
    """Draw pose skeleton with imbalance highlighting"""
    h, w = image.shape[:2]
    connections = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 12), (11, 23), (12, 24), (23, 24)
    ]

    highlight_red = set()
    if features:
        shoulder_diff       = features.get('shoulder_height_diff', 0)
        shoulder_angle_diff = features.get('shoulder_angle_diff', 0)
        if shoulder_diff > 0.05 or shoulder_angle_diff > 15:
            highlight_red.add(11)
            highlight_red.add(12)

        wrist_diff = features.get('wrist_height_diff', 0)
        if wrist_diff > 0.05:
            highlight_red.add(15)
            highlight_red.add(16)

        elbow_diff       = features.get('elbow_height_diff', 0)
        elbow_angle_diff = features.get('elbow_angle_diff', 0)
        if elbow_diff > 0.05 or elbow_angle_diff > 15:
            highlight_red.add(13)
            highlight_red.add(14)

        hip_diff = features.get('hip_height_diff', 0)
        if hip_diff > 0.05 and features.get('has_hip_data', False):
            highlight_red.add(23)
            highlight_red.add(24)

    for start_i, end_i in connections:
        start      = landmarks[0][start_i]
        end        = landmarks[0][end_i]
        line_color = (0, 0, 255) if (start_i in highlight_red or end_i in highlight_red) else (0, 255, 255)
        cv2.line(image,
                 (int(start.x * w), int(start.y * h)),
                 (int(end.x * w),   int(end.y * h)),
                 line_color, 3)

    for idx, lm in enumerate(landmarks[0]):
        x, y = int(lm.x * w), int(lm.y * h)
        if idx in highlight_red:
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
            cv2.circle(image, (x, y), 13, (0, 0, 180), 2)
            cv2.circle(image, (x, y), 16, (255, 255, 255), 1)
        else:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(image, (x, y), 7, (255, 255, 255), 2)

    return image

def draw_bar_path_indicator(frame, features):
    """Draw bar path visualization"""
    h, w = frame.shape[:2]

    left_wrist_x  = int(features['left_wrist_x'] * w)
    left_wrist_y  = int(features['left_wrist_y'] * h)
    right_wrist_x = int(features['right_wrist_x'] * w)
    right_wrist_y = int(features['right_wrist_y'] * h)

    bar_vertical_tilt = features['bar_vertical_tilt']

    if bar_vertical_tilt > 0.05:
        color  = (0, 0, 255)
        status = "UNEVEN"
    else:
        color  = (0, 255, 0)
        status = "LEVEL"

    cv2.line(frame, (left_wrist_x, left_wrist_y), (right_wrist_x, right_wrist_y), color, 6)
    cv2.circle(frame, (left_wrist_x, left_wrist_y), 12, color, -1)
    cv2.circle(frame, (right_wrist_x, right_wrist_y), 12, color, -1)
    cv2.circle(frame, (left_wrist_x, left_wrist_y), 15, (255, 255, 255), 2)
    cv2.circle(frame, (right_wrist_x, right_wrist_y), 15, (255, 255, 255), 2)

    center_x = int(w / 2)
    cv2.line(frame, (center_x, 0), (center_x, h), (128, 128, 128), 1)

    bar_center_pixel  = int((left_wrist_x + right_wrist_x) / 2)
    bar_center_text_y = max(50, min(left_wrist_y, right_wrist_y) - 30)

    cv2.putText(frame, f"BAR: {status}", (bar_center_pixel - 50, bar_center_text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

def draw_symmetry_indicators(frame, features):
    """Draw symmetry analysis panel"""
    h, w = frame.shape[:2]

    symmetry_score = features.get('symmetry_score', 0)
    shoulder_diff  = features.get('shoulder_height_diff', 0) * 100
    elbow_diff     = features.get('elbow_height_diff', 0) * 100
    wrist_diff     = features.get('wrist_height_diff', 0) * 100
    grade          = features.get('overall_symmetry_grade', 'C')

    grade_color = {
        'A': (0, 255, 0),
        'B': (0, 255, 255),
        'C': (0, 165, 255),
        'D': (0, 0, 255),
    }.get(grade, (0, 165, 255))

    panel_x = w - 260
    has_hip  = features.get('has_hip_data', False)
    panel_h  = 230 if has_hip else 200

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 10), (w - 10, panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    y = 40
    cv2.putText(frame, "SYMMETRY", (panel_x + 10, y),
               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    y += 35
    cv2.putText(frame, f"Grade: {grade}", (panel_x + 10, y),
               cv2.FONT_HERSHEY_DUPLEX, 1.0, grade_color, 2)
    y += 30
    cv2.putText(frame, f"Score: {symmetry_score:.3f}", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 30
    cv2.putText(frame, f"Shoulders: {shoulder_diff:.1f}%", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if shoulder_diff < 5 else (0, 0, 255), 1)
    y += 22
    cv2.putText(frame, f"Elbows: {elbow_diff:.1f}%", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if elbow_diff < 5 else (0, 0, 255), 1)
    y += 22
    cv2.putText(frame, f"Wrists: {wrist_diff:.1f}%", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if wrist_diff < 5 else (0, 0, 255), 1)

    if has_hip:
        hip_diff = features.get('hip_height_diff', 0) * 100
        y += 22
        cv2.putText(frame, f"Hips: {hip_diff:.1f}%", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if hip_diff < 5 else (0, 0, 255), 1)

    return frame

def draw_elbow_angle_indicator(frame, features, current_phase):
    """Draw elbow angle information panel"""
    h, w = frame.shape[:2]

    avg_elbow   = features.get('avg_elbow_angle', 0)
    left_elbow  = features.get('left_elbow_angle', 0)
    right_elbow = features.get('right_elbow_angle', 0)
    has_hip     = features.get('has_hip_data', False)

    panel_y = 250 if has_hip else 220
    panel_x = w - 260

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    y = panel_y + 30
    cv2.putText(frame, "ELBOW ANGLE", (panel_x + 10, y),
               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

    y += 30
    if avg_elbow > 140:
        angle_color = (0, 255, 0)
    elif avg_elbow < 80:
        angle_color = (0, 165, 255)
    else:
        angle_color = (0, 255, 255)

    cv2.putText(frame, f"Avg: {avg_elbow:.1f}", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, angle_color, 2)
    y += 25
    cv2.putText(frame, f"Left:  {left_elbow:.1f}", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 20
    cv2.putText(frame, f"Right: {right_elbow:.1f}", (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 25

    phase_color = {
        "TOP":         (0, 255, 0),
        "BOTTOM":      (0, 165, 255),
        "MOVING_UP":   (255, 200, 0),
        "MOVING_DOWN": (0, 255, 255),
        "UNKNOWN":     (128, 128, 128),
    }.get(current_phase, (128, 128, 128))

    cv2.putText(frame, current_phase, (panel_x + 10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)

    return frame

def draw_rep_counter(frame, rep_counter):
    """Draw rep counter display"""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - 150, 10), (w//2 + 150, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    cv2.putText(frame, f"REPS: {rep_counter.rep_count}", (w//2 - 80, 50),
               cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

    phase_color = {
        "TOP":         (0, 255, 0),
        "MOVING_DOWN": (0, 255, 255),
        "BOTTOM":      (0, 165, 255),
        "MOVING_UP":   (255, 200, 0),
        "UNKNOWN":     (128, 128, 128),
    }.get(rep_counter.current_phase, (128, 128, 128))

    cv2.putText(frame, rep_counter.current_phase, (w//2 - 70, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)

    return frame

def draw_rep_history(frame, rep_counter):
    """Draw recent rep history"""
    h, w = frame.shape[:2]

    history = rep_counter.get_rep_history(5)
    if not history:
        return frame

    panel_width  = 500
    panel_height = 100
    panel_x      = w//2 - panel_width//2
    panel_y      = h - panel_height - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, h - 10), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    cv2.putText(frame, "RECENT REPS:", (panel_x + 10, panel_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    x_offset = panel_x + 20
    y = panel_y + 60

    for rep in reversed(history):
        grade = rep.get_quality_grade()
        color = rep.get_quality_color()

        cv2.putText(frame, f"#{rep.rep_number}", (x_offset, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.circle(frame, (x_offset + 25, y - 5), 15, color, -1)
        cv2.circle(frame, (x_offset + 25, y - 5), 17, (255, 255, 255), 2)
        cv2.putText(frame, grade, (x_offset + 19, y),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)

        x_offset += 90

    return frame

def draw_rep_notification(frame, rep_analysis):
    """Draw rep completion notification"""
    if rep_analysis is None:
        return frame

    h, w = frame.shape[:2]
    grade = rep_analysis.get_quality_grade()
    color = rep_analysis.get_quality_color()

    overlay = frame.copy()
    cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)

    cv2.putText(frame, f"REP #{rep_analysis.rep_number} COMPLETE",
               (w//4 + 50, h//3 + 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Grade: {grade}",
               (w//4 + 120, h//3 + 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

    technique_text  = "CORRECT" if rep_analysis.prediction == 1 else "INCORRECT"
    technique_color = (0, 255, 0) if rep_analysis.prediction == 1 else (0, 0, 255)
    cv2.putText(frame, technique_text,
               (w//4 + 140, h//3 + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, technique_color, 2)
    cv2.putText(frame, f"{rep_analysis.rep_duration:.1f}s",
               (w//4 + 160, h//3 + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame

def draw_enhanced_highlights(frame, landmarks, features, prediction):
    """Draw warning highlights for incorrect form"""
    if prediction == 1:
        return frame

    h, w = frame.shape[:2]
    avg_elbow = features.get('avg_elbow_angle', 0)

    if avg_elbow > 120:
        color      = (0, 0, 255)
        radius     = 50
        label_text = "CRITICAL"
    elif avg_elbow > 75:
        color      = (0, 140, 255)
        radius     = 40
        label_text = "WARNING"
    else:
        return frame

    overlay    = frame.copy()
    left_elbow = landmarks[0][13]
    right_elbow = landmarks[0][14]
    left_pos   = (int(left_elbow.x * w), int(left_elbow.y * h))
    right_pos  = (int(right_elbow.x * w), int(right_elbow.y * h))

    cv2.circle(overlay, left_pos, radius, color, -1)
    cv2.circle(overlay, right_pos, radius, color, -1)
    cv2.circle(overlay, left_pos, radius + 5, (255, 255, 255), 3)
    cv2.circle(overlay, right_pos, radius + 5, (255, 255, 255), 3)

    frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    cv2.putText(frame, label_text, (left_pos[0] - 40, left_pos[1] - 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame

def draw_info_panel(frame, prediction, confidence, fps, errors, exercise_name):
    """Draw main information panel"""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (550, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    cv2.putText(frame, f"{exercise_name}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

    if prediction == 1:
        label  = "CORRECT TECHNIQUE"
        symbol = "[OK]"
        color  = (0, 255, 0)
    else:
        label  = "INCORRECT TECHNIQUE"
        symbol = "[!]"
        color  = (0, 0, 255)

    cv2.putText(frame, f"{symbol} {label}", (20, 70),
               cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 3)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 115),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if errors and prediction == 0:
        severity  = errors[0].severity
        sev_color = (0, 0, 255) if severity == "CRITICAL" else (0, 165, 255)
        cv2.putText(frame, f"{len(errors)} Issue(s) - {severity}", (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, sev_color, 2)
    elif prediction == 1:
        cv2.putText(frame, "No issues detected", (20, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Heatmap status
    heatmap_status = "ON" if SYSTEM_STATE['heatmap_enabled'] else "OFF"
    heatmap_color = (0, 255, 0) if SYSTEM_STATE['heatmap_enabled'] else (128, 128, 128)
    cv2.putText(frame, f"Heatmap: {heatmap_status}", (w - 150, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, heatmap_color, 1)

    cv2.putText(frame, "Q:Quit | R:Report | H:Heatmap | A:Audio", (20, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame

def get_angle_context(error, features):
    avg_elbow = features.get('avg_elbow_angle', 0)
    bar_tilt  = features.get('bar_vertical_tilt', 0) * 100
    sh_diff   = features.get('shoulder_height_diff', 0) * 100
    el_diff   = features.get('elbow_angle_diff', 0)
    contexts = {
        'Extreme Elbow Flare':   f"{avg_elbow:.1f}deg > 120deg threshold",
        'Excessive Elbow Flare': f"{avg_elbow:.1f}deg > 75deg threshold",
        'Uneven Bar Path':       f"Tilt: {bar_tilt:.1f}% > 5% threshold",
        'Shoulder Height':       f"Diff: {sh_diff:.1f}% > 5% threshold",
        'Elbow Height':          f"Angle diff: {el_diff:.1f}deg > 15deg",
        'Grip Too Wide':         f"Forearms angled outward > 20deg",
        'Grip Too Narrow':       f"Forearms angled inward > 20deg",
        'Bar Path Off':          f"Off-centre detected",
        'Hip Tilt':              f"Hip height imbalance > 5%",
        'Wrist Height':          f"Wrist height diff > 5%",
        'Scapular':              f"Retraction offset > 15%",
    }
    for key, val in contexts.items():
        if key.lower() in error.error_type.lower():
            return val
    return ""

def draw_medical_warning(frame, errors, prediction, features=None):
    """Draw medical warning panel with specific angle values and impact explanation"""
    if not errors or prediction == 1:
        return frame
    h, w = frame.shape[:2]
    error      = errors[0]
    box_height = 250
    box_y      = h - box_height - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, box_y), (w - 10, h - 10), (0, 0, 80), -1)
    frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
    y = box_y + 28

    # Error type
    cv2.putText(frame, f"[!] {error.error_type}", (20, y),
               cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    y += 30

    # Specific measured value vs threshold
    if features:
        ctx = get_angle_context(error, features)
        if ctx:
            cv2.putText(frame, f"    Measured: {ctx}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1)
            y += 24

    # Injury impact
    if error.injury_risks:
        cv2.putText(frame, f"  Risk: {error.injury_risks[0][:58]}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        y += 22

    # Correction cue
    if error.correction_steps:
        step_idx = 0 if error.severity == "CRITICAL" else 1 if len(error.correction_steps) > 1 else 0
        cue = error.correction_steps[step_idx]
        cv2.putText(frame, f"  Fix: {cue[:60]}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 1)
        y += 24

    # Affected muscles
    cv2.putText(frame, "AFFECTED MUSCLES:", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
    y += 22
    for muscle in error.affected_muscles[:2]:
        cv2.putText(frame, f"  * {muscle.name[:35]}", (30, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

    y += 5
    if error.severity == "CRITICAL":
        cv2.putText(frame, "CRITICAL INJURY RISK", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"INJURY RISK: {error.severity}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 1)
    return frame

def draw_audio_indicator(frame, audio_coach, audio_enabled):
    """Draw audio status indicator"""
    h, w = frame.shape[:2]
    x, y = 20, h - 60

    if audio_enabled:
        if audio_coach.is_speaking:
            color = (0, 255, 0)
            text  = "SPEAKING..."
        else:
            color = (0, 255, 255)
            text  = "AUDIO ON"
    else:
        color = (128, 128, 128)
        text  = "AUDIO OFF"

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 25), (x + 180, y + 5), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def main():
    """Main application loop"""
    print("\nREAL-TIME TECHNIQUE ANALYSIS")
    print("Multi-Exercise Support with Medical Feedback")
    print("-" * 50)

    # Exercise selection
    exercise_type, exercise_config = select_exercise()
    exercise_name = exercise_config['name']

    # Pose detector setup
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # Person detection filter
    print("\nInitializing person detection filter...")
    try:
        person_isolator = PersonIsolator(method='yolo')
        print("Person detection filter active (YOLOv8)")
    except Exception as e:
        print(f"Person filter initialization failed: {e}")
        print("Continuing without person filter")
        person_isolator = None

    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # System components
    rep_counter   = RepCounter(history_size=100)
    report_viewer = ReportViewer()
    audio_coach   = AudioCoach(rate=160, volume=0.9)

    # Audio setup
    print("\nAudio Coaching Setup")
    print("-" * 50)
    enable_audio_input = input("Enable voice coaching? (y/n): ").lower().strip()
    audio_enabled = enable_audio_input == 'y'

    if audio_enabled:
        print("Audio coaching enabled")
        voice_choice = input("Voice preference (0=default, 1=alternate): ").strip()
        if voice_choice == '1':
            audio_coach.change_voice(1)
        freq_choice = input("Feedback frequency (1=frequent, 2=normal, 3=infrequent): ").strip()
        if freq_choice == '1':
            audio_coach.set_cooldown(2.0)
        elif freq_choice == '3':
            audio_coach.set_cooldown(5.0)
    else:
        print("Audio disabled (press A to enable)")

    print("\nSystem Ready")
    print("-" * 50)
    print(f"Exercise: {exercise_name}")
    print(f"Audio: {'Enabled' if audio_enabled else 'Disabled'}")
    print(f"Heatmap Visualization: {'Enabled' if SYSTEM_STATE['heatmap_enabled'] else 'Disabled'}")
    print(f"Person Detection Filter: {'Enabled' if person_isolator and SYSTEM_STATE['person_filter_enabled'] else 'Disabled'}")
    print(f"Auto-save heatmaps: Grades C, D, F")
    print(f"Rep Detection: Elbow-based (Middle to Bottom to Up)")
    print(f"Camera setup: {exercise_config['camera_position']}")
    print("\nControls:")
    print("  Q/ESC  - Quit")
    print("  R      - Open/Close report (current form analysis)")
    print("  S      - Set summary (view all bad rep heatmaps)")
    print("  H      - Toggle heatmap visualization")
    print("  P      - Toggle person detection filter")
    print("  A      - Toggle audio")
    print("  (In report) UP/DOWN or J/K to scroll, V:View heatmaps, S:Save")
    print("\nTip: Press 'S' after your set to see all captured heatmaps")
    print("Tip: If landmarks are misplaced, press 'P' to toggle person filter")
    print("\nStarting...\n")

    # State variables
    prev_time              = time.time()
    fps                    = 0
    current_errors         = []
    current_prediction     = 1
    current_features       = {}
    current_confidence     = 0.0
    last_rep_notification  = None
    notification_time      = 0
    bad_rep_heatmaps       = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Pose detection with person filter
            timestamp_ms = int(time.time() * 1000)
            
            if SYSTEM_STATE['person_filter_enabled'] and person_isolator:
                detection_result, person_bbox = integrate_person_filter(
                    frame, detector, person_isolator, timestamp_ms
                )
                
                if person_bbox and not report_viewer.active:
                    x, y, w_box, h_box = person_bbox
                    cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                    cv2.putText(frame, "Person ROI", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                person_bbox = None

            # Calculate FPS
            curr_time = time.time()
            fps       = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Process pose detection
            if detection_result.pose_landmarks:
                try:
                    is_valid, missing = ExerciseConfig.check_landmark_visibility(
                        detection_result.pose_landmarks,
                        exercise_type
                    )

                    if not is_valid:
                        frame = draw_skeleton(frame, detection_result.pose_landmarks, features)
                        if not report_viewer.active:
                            cv2.putText(frame, f"Missing: {', '.join(missing)}", (20, 100),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            cv2.putText(frame, "Adjust camera position", (20, 130),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        # Full analysis
                        features           = extract_features(detection_result.pose_landmarks, exercise_config)
                        current_features   = features.copy()
                        prediction, confidence = predict_technique(features)
                        current_prediction = prediction
                        current_confidence = confidence
                        current_errors     = anatomical_expert.analyze_technique(features, exercise_config)

                        # Audio feedback
                        if audio_enabled and not report_viewer.active:
                            audio_coach.analyze_and_coach(
                                features, prediction, current_errors, rep_counter.rep_count
                            )

                        # Rep counting
                        rep_completed, rep_analysis = rep_counter.update(
                            features, prediction, confidence, current_errors
                        )

                        if rep_completed:
                            last_rep_notification = rep_analysis
                            notification_time     = time.time()
                            grade = rep_analysis.get_quality_grade()
                            print(f"Rep #{rep_analysis.rep_number} - Grade: {grade} - {rep_analysis.rep_duration:.1f}s")
                            
                            # Auto-save heatmap for poor grades
                            # Use grade-based prediction override (not ML prediction)
                            # so heatmap generates even when ML says CORRECT but grade is bad
                            heatmap_prediction = 0 if grade in ['C', 'D', 'F'] else prediction
                            if grade in ['C', 'D', 'F'] and detection_result.pose_landmarks:
                                try:
                                    heatmap_viz = heatmap_generator.generate_report_visualization(
                                        frame=frame.copy(),
                                        landmarks=detection_result.pose_landmarks,
                                        prediction=heatmap_prediction,
                                        features=features,
                                        errors=current_errors if current_errors else rep_analysis.errors
                                    )
                                    
                                    bad_rep_heatmaps.append({
                                        'rep_number': rep_analysis.rep_number,
                                        'grade': grade,
                                        'image': heatmap_viz,
                                        'timestamp': time.time(),
                                        'duration': rep_analysis.rep_duration
                                    })
                                    print(f"  Heatmap captured for Rep #{rep_analysis.rep_number} (Grade {grade})")
                                except Exception as e:
                                    print(f"  Warning: Could not generate heatmap: {e}")
                            
                            if audio_enabled:
                                audio_coach.rep_completed_feedback(rep_analysis)

                        # Draw visualizations
                        frame = draw_skeleton(frame, detection_result.pose_landmarks, features)
                        frame = draw_bar_path_indicator(frame, features)
                        frame = draw_symmetry_indicators(frame, features)
                        frame = draw_elbow_angle_indicator(frame, features, rep_counter.current_phase)
                        frame = draw_rep_counter(frame, rep_counter)
                        frame = draw_rep_history(frame, rep_counter)

                        if last_rep_notification and (time.time() - notification_time < 2.0):
                            frame = draw_rep_notification(frame, last_rep_notification)

                        frame = draw_enhanced_highlights(frame, detection_result.pose_landmarks,
                                                        features, prediction)
                        frame = draw_anatomical_overlay(frame, detection_result.pose_landmarks,
                                                       current_errors)

                        # Generate heatmap overlay
                        if SYSTEM_STATE['heatmap_enabled'] and prediction == 0 and not report_viewer.active:
                            frame = heatmap_generator.generate_heatmap_overlay(
                                frame=frame,
                                landmarks=detection_result.pose_landmarks,
                                prediction=prediction,
                                features=features,
                                errors=current_errors
                            )

                        # Info panels
                        if not report_viewer.active:
                            frame = draw_info_panel(frame, prediction, confidence, fps,
                                                    current_errors, exercise_name)
                            frame = draw_medical_warning(frame, current_errors, prediction, features)
                            frame = draw_audio_indicator(frame, audio_coach, audio_enabled)

                except Exception as e:
                    if not report_viewer.active:
                        cv2.putText(frame, f"Error: {str(e)[:40]}", (20, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            else:
                if not report_viewer.active:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (w - 10, 150), (50, 50, 50), -1)
                    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                    cv2.putText(frame, "No person detected", (20, 60),
                               cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 165, 255), 3)
                    frame = draw_rep_counter(frame, rep_counter)
                    frame = draw_audio_indicator(frame, audio_coach, audio_enabled)

            # Draw report overlay
            if report_viewer.active:
                frame = report_viewer.draw(frame)

            # Display
            cv2.imshow('Medical-Grade Exercise Analysis', frame)

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF

            if report_viewer.active:
                # Report viewer controls
                if key == 27 or key == ord('r'):
                    report_viewer.hide()
                    print("Report closed")
                elif key in [82, 0, 2, 63232]:
                    report_viewer.scroll_up()
                elif key in [84, 1, 3, 63233]:
                    report_viewer.scroll_down()
                elif key == ord('k'):
                    report_viewer.scroll_up()
                elif key == ord('j'):
                    report_viewer.scroll_down()
                elif key == ord('s'):
                    timestamp   = int(time.time())
                    is_set      = "SET" in report_viewer.report_title.upper()
                    report_file = f'{"set" if is_set else "analysis"}_report_{timestamp}.txt'
                    
                    report_text = report_viewer.report_text
                    if is_set and report_viewer.rep_heatmaps:
                        report_text += "\n\n" + "-"*70
                        report_text += "\nSAVED FILES"
                        report_text += "\n" + "-"*70
                        report_text += f"\n\nReport file: {report_file}"
                        report_text += "\n\nIndividual heatmap files:"
                        for hm in report_viewer.rep_heatmaps:
                            filename = f"rep_{hm['rep_number']:02d}_grade_{hm['grade']}_heatmap_{timestamp}.jpg"
                            report_text += f"\n  - {filename}"
                        report_text += f"\n\nCombined heatmap: set_summary_heatmaps_{timestamp}.jpg"
                        report_text += "\n\nAll files saved in current directory."
                    
                    with open(report_file, 'w') as f:
                        f.write(report_text)
                    print(f"Saved text report: {report_file}")
                    
                    if report_viewer.heatmap_image is not None:
                        heatmap_file = f'heatmap_visualization_{timestamp}.jpg'
                        if report_viewer.save_heatmap(heatmap_file):
                            print(f"Saved heatmap: {heatmap_file}")
                    
                    if report_viewer.rep_heatmaps:
                        saved_files = report_viewer.save_all_rep_heatmaps(timestamp)
                        if saved_files:
                            print(f"Saved {len(saved_files)} rep heatmap(s):")
                            for f in saved_files:
                                print(f"  - {f}")
                        
                        try:
                            heatmaps = [hm['image'] for hm in report_viewer.rep_heatmaps]
                            if len(heatmaps) > 0:
                                if len(heatmaps) <= 3:
                                    combined = np.vstack(heatmaps)
                                else:
                                    mid = (len(heatmaps) + 1) // 2
                                    row1 = np.hstack(heatmaps[:mid])
                                    row2 = np.hstack(heatmaps[mid:])
                                    if row1.shape[1] != row2.shape[1]:
                                        diff = abs(row1.shape[1] - row2.shape[1])
                                        if row1.shape[1] > row2.shape[1]:
                                            pad = np.zeros((row2.shape[0], diff, 3), dtype=np.uint8)
                                            row2 = np.hstack([row2, pad])
                                        else:
                                            pad = np.zeros((row1.shape[0], diff, 3), dtype=np.uint8)
                                            row1 = np.hstack([row1, pad])
                                    combined = np.vstack([row1, row2])
                                
                                combined_file = f'set_summary_heatmaps_{timestamp}.jpg'
                                cv2.imwrite(combined_file, combined)
                                print(f"Saved combined heatmap: {combined_file}")
                        except Exception as e:
                            print(f"Note: Could not create combined image: {e}")
                            
                elif key == ord('v'):
                    if report_viewer.heatmap_image is not None:
                        cv2.imshow('Heatmap Visualization', report_viewer.heatmap_image)
                        print("Heatmap displayed - press any key to close")
                    elif report_viewer.rep_heatmaps:
                        print(f"\nDisplaying {len(report_viewer.rep_heatmaps)} heatmap(s)...")
                        for i, hm in enumerate(report_viewer.rep_heatmaps):
                            window_name = f"Rep #{hm['rep_number']} - Grade {hm['grade']} Heatmap"
                            cv2.imshow(window_name, hm['image'])
                            print(f"  {i+1}. {window_name}")
                        print("\nPress any key to close all heatmap windows")
                    else:
                        print("No heatmap available (only for incorrect form or bad reps)")

            else:
                # Main interface controls
                if key == ord('q') or key == 27:
                    break

                elif key == ord('a'):
                    audio_enabled = not audio_enabled
                    status = 'enabled' if audio_enabled else 'disabled'
                    print(f"Audio {status}")
                    if audio_enabled and audio_coach.tts_available:
                        audio_coach.speak_async(f"Audio coaching {status}")

                elif key == ord('h') or key == ord('H'):
                    SYSTEM_STATE['heatmap_enabled'] = not SYSTEM_STATE['heatmap_enabled']
                    status = "ON" if SYSTEM_STATE['heatmap_enabled'] else "OFF"
                    print(f"Heatmap visualization: {status}")

                elif key == ord('p') or key == ord('P'):
                    SYSTEM_STATE['person_filter_enabled'] = not SYSTEM_STATE['person_filter_enabled']
                    status = "ON" if SYSTEM_STATE['person_filter_enabled'] else "OFF"
                    print(f"Person detection filter: {status}")
                    if SYSTEM_STATE['person_filter_enabled'] and person_isolator:
                        print("  Landmarks should now be more accurate in cluttered environments")
                    else:
                        print("  Using standard pose detection (may confuse equipment as body)")

                elif key == ord('r'):
                    if current_errors:
                        report = anatomical_expert.generate_detailed_report(current_errors)
                        report_viewer.show(
                            report, 
                            "Medical Analysis - Errors Detected",
                            frame=frame.copy() if detection_result.pose_landmarks else None,
                            landmarks=detection_result.pose_landmarks if detection_result.pose_landmarks else None,
                            prediction=current_prediction,
                            features=current_features,
                            errors=current_errors
                        )
                        print("\nMedical report displayed")
                        if current_prediction == 0:
                            print("  Press V to view heatmap | S to save report + heatmap")
                    elif current_features:
                        report = generate_current_analysis_report(
                            current_features, current_prediction,
                            current_confidence, rep_counter.get_last_rep(), exercise_name
                        )
                        report_viewer.show(
                            report, 
                            "Technique Analysis",
                            frame=frame.copy() if detection_result.pose_landmarks else None,
                            landmarks=detection_result.pose_landmarks if detection_result.pose_landmarks else None,
                            prediction=current_prediction,
                            features=current_features,
                            errors=current_errors
                        )
                        print("\nAnalysis displayed")
                        if current_prediction == 0:
                            print("  Press V to view heatmap | S to save report + heatmap")
                    else:
                        report_viewer.show(
                            "No analysis data yet.\n\nPlease position yourself in front of the camera and begin exercising.\n\nPress R or ESC to close.",
                            "Waiting for Data"
                        )
                        print("\nNo data yet - position yourself in front of camera")

                    if report_viewer.active:
                        if current_prediction == 0:
                            print("  UP/DOWN or J/K to scroll | V:View heatmap | S:Save all | R/ESC:Close")
                        else:
                            print("  UP/DOWN or J/K to scroll | S:Save report | R/ESC:Close")

                elif key == ord('s'):
                    if rep_counter.rep_count > 0:
                        report = rep_counter.generate_set_report()
                        
                        combined_heatmap_file = None
                        if bad_rep_heatmaps:
                            try:
                                heatmaps = [hm['image'] for hm in bad_rep_heatmaps]
                                
                                if len(heatmaps) == 1:
                                    combined_heatmap = heatmaps[0]
                                elif len(heatmaps) <= 3:
                                    combined_heatmap = np.vstack(heatmaps)
                                else:
                                    mid = (len(heatmaps) + 1) // 2
                                    row1 = np.hstack(heatmaps[:mid])
                                    row2 = np.hstack(heatmaps[mid:])
                                    
                                    if row1.shape[1] != row2.shape[1]:
                                        diff = abs(row1.shape[1] - row2.shape[1])
                                        if row1.shape[1] > row2.shape[1]:
                                            pad = np.zeros((row2.shape[0], diff, 3), dtype=np.uint8)
                                            row2 = np.hstack([row2, pad])
                                        else:
                                            pad = np.zeros((row1.shape[0], diff, 3), dtype=np.uint8)
                                            row1 = np.hstack([row1, pad])
                                    
                                    combined_heatmap = np.vstack([row1, row2])
                                
                                timestamp = int(time.time())
                                combined_heatmap_file = f'set_heatmap_combined_{timestamp}.jpg'
                                cv2.imwrite(combined_heatmap_file, combined_heatmap)
                                
                            except Exception as e:
                                print(f"Warning: Could not create combined heatmap: {e}")
                        
                        if bad_rep_heatmaps:
                            report += "\n\n" + "-"*70
                            report += "\nGRAD-CAM HEATMAP VISUALIZATION"
                            report += "\n" + "-"*70
                            report += f"\n\nCaptured {len(bad_rep_heatmaps)} heatmap(s) for poor form reps (Grade C/D/F)"
                            
                            if combined_heatmap_file:
                                report += f"\n\nCOMBINED HEATMAP IMAGE:"
                                report += f"\n  File: {combined_heatmap_file}"
                                report += f"\n  Visualization: 3-panel format (Original | Heatmap | Overlay)"
                                report += f"\n  Color Scale: Blue (safe) to Red (critical injury risk)"
                            
                            report += "\n\nINDIVIDUAL REP HEATMAPS:"
                            for hm in bad_rep_heatmaps:
                                report += f"\n  - Rep #{hm['rep_number']}: Grade {hm['grade']} - {hm['duration']:.1f}s"
                            
                            report += "\n\nHEATMAP INTERPRETATION:"
                            report += "\n  Red/Orange zones    = Critical form errors (high injury risk)"
                            report += "\n  Yellow zones        = Moderate attention needed"
                            report += "\n  Green/Blue zones    = Acceptable form"
                            
                            report += "\n\nNOTE: Heatmaps use Random Forest feature importance + biomechanical"
                            report += "\n      deviation to highlight problematic body regions. This provides"
                            report += "\n      Grad-CAM style visualization while maintaining model explainability."
                            
                            report += "\n\nPress V to view heatmaps | Press S to save report + all images"
                        
                        # Add padding so user can scroll to see heatmaps
                        # Add padding so user can scroll to see heatmaps
                        if bad_rep_heatmaps:
                            report += chr(10) * 25
                        report_viewer.show(report, "Set Summary", rep_heatmaps=bad_rep_heatmaps)
                        if bad_rep_heatmaps:
                            print(f"  Found {len(bad_rep_heatmaps)} rep(s) with heatmaps")
                            if combined_heatmap_file:
                                print(f"  Combined heatmap saved: {combined_heatmap_file}")
                            print("  Press V to view heatmaps | S to save all")
                        else:
                            print("  UP/DOWN or J/K to scroll | R or ESC to close | S to save")
                    else:
                        print("\nNo reps completed yet")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if rep_counter.rep_count > 0:
            print("\nSession Complete")
            print("-" * 50)
            summary = rep_counter.get_set_summary()

            if audio_enabled:
                audio_coach.set_completed_feedback(summary)

            print(f"\nTotal Reps:    {summary['total_reps']}")
            print(f"Correct:       {summary['correct_reps']} ({summary['accuracy_rate']:.1f}%)")
            print(f"Avg Confidence:{summary['average_confidence']*100:.1f}%")
            print(f"Avg Symmetry:  {summary['average_symmetry_score']:.3f}")
            print("\nGrade Distribution:")
            for grade, count in summary['grade_distribution'].items():
                if count > 0:
                    print(f"  {grade}: {count} rep(s)")
            
            # Auto-save heatmaps
            if bad_rep_heatmaps:
                print(f"\nAuto-saving {len(bad_rep_heatmaps)} heatmap(s) from poor form reps...")
                timestamp = int(time.time())
                saved_count = 0
                
                for hm in bad_rep_heatmaps:
                    filename = f"rep_{hm['rep_number']:02d}_grade_{hm['grade']}_heatmap_{timestamp}.jpg"
                    try:
                        cv2.imwrite(filename, hm['image'])
                        print(f"  Rep #{hm['rep_number']} (Grade {hm['grade']}) saved to {filename}")
                        saved_count += 1
                    except Exception as e:
                        print(f"  Failed to save Rep #{hm['rep_number']}: {e}")
                
                # Create combined summary
                try:
                    if len(bad_rep_heatmaps) > 0:
                        heatmaps = [hm['image'] for hm in bad_rep_heatmaps]
                        if len(heatmaps) <= 3:
                            combined = np.vstack(heatmaps)
                        else:
                            mid = (len(heatmaps) + 1) // 2
                            row1 = np.hstack(heatmaps[:mid])
                            row2 = np.hstack(heatmaps[mid:])
                            if row1.shape[1] != row2.shape[1]:
                                diff = abs(row1.shape[1] - row2.shape[1])
                                if row1.shape[1] > row2.shape[1]:
                                    pad = np.zeros((row2.shape[0], diff, 3), dtype=np.uint8)
                                    row2 = np.hstack([row2, pad])
                                else:
                                    pad = np.zeros((row1.shape[0], diff, 3), dtype=np.uint8)
                                    row1 = np.hstack([row1, pad])
                            combined = np.vstack([row1, row2])
                        
                        combined_file = f'session_heatmaps_combined_{timestamp}.jpg'
                        cv2.imwrite(combined_file, combined)
                        print(f"  Combined heatmap saved to {combined_file}")
                except Exception as e:
                    print(f"  Note: Could not create combined image: {e}")
                
                print(f"\nTotal: {saved_count} heatmap file(s) saved")
            else:
                print("\nNo poor form reps detected - all grades were A or B")
            
            print("-" * 50)

        print("\nSession complete\n")

if __name__ == "__main__":
    main()