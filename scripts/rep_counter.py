from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import deque
import time

# Import improved grading system
from improved_grading import calculate_injury_risk_score, get_grade_from_risk_score, get_improved_grade_with_explanation, apply_biomechanical_overrides

@dataclass
class RepAnalysis:
    """
    Complete analysis data for a single rep
    """
    rep_number: int
    timestamp: float
    prediction: int  # 0 = incorrect, 1 = correct
    confidence: float
    errors: List  # List of BiomechanicalError objects
    features: Dict
    symmetry_grade: str
    symmetry_score: float
    bar_tilt_max: float
    elbow_flare_max: float
    elbow_angle_min: float  # Minimum elbow angle reached
    elbow_angle_max: float  # Maximum elbow angle reached
    rep_duration: float  # Time taken for rep
    
    def get_quality_grade(self) -> str:
        """
        Calculate overall quality grade using biomechanical overrides + injury risk scoring
        Biomechanical rules take priority over ML prediction.
        """
        _, forced_grade, _ = apply_biomechanical_overrides(self.features, self.errors, self.prediction)
        if forced_grade:
            return forced_grade
        risk_score = calculate_injury_risk_score(self.features, self.errors)
        grade = get_grade_from_risk_score(risk_score, self.prediction, self.confidence)
        return grade
    
    def get_risk_score(self) -> float:
        """Get numerical injury risk score (0-100)"""
        return calculate_injury_risk_score(self.features, self.errors)
    
    def get_grade_explanation(self) -> List[str]:
        """Get detailed explanation of grade"""
        _, _, explanation = get_improved_grade_with_explanation(
            self.features, self.prediction, self.confidence, self.errors
        )
        return explanation
    
    def get_quality_color(self):
        """
        Get BGR color for quality grade - based on injury risk
        """
        risk_score = self.get_risk_score()
        
        if risk_score >= 70:
            return (0, 0, 255)      # Red - CRITICAL
        elif risk_score >= 50:
            return (0, 100, 255)    # Orange - SEVERE
        elif risk_score >= 30:
            return (0, 200, 255)    # Yellow - MODERATE
        elif risk_score >= 15:
            return (0, 255, 200)    # Yellow-Green - MINOR
        else:
            return (0, 255, 0)      # Green - GOOD

class RepCounter:
    """
    Improved rep counter - starts counting from ANY position
    Counts each complete concentric (pushing) phase
    """
    
    def __init__(self, history_size=10):
        """
        Initialize rep counter
        
        Args:
            history_size: Number of past reps to store
        """
        self.rep_count = 0
        self.history_size = history_size
        self.rep_history = deque(maxlen=history_size)
        
        # THRESHOLDS - Adjust these based on exercise
        self.EXTENDED_THRESHOLD = 140    # Elbow extended (top position)
        self.FLEXED_THRESHOLD = 80       # Elbow flexed (bottom position)
        self.MOVEMENT_THRESHOLD = 15     # Minimum angle change to detect movement
        
        # Phase tracking
        self.current_phase = "UNKNOWN"
        self.previous_phase = "UNKNOWN"
        
        # Elbow angle tracking
        self.elbow_angle_history = deque(maxlen=10)
        
        # Rep detection state
        self.rep_in_progress = False
        self.reached_bottom = False      # Track if we went to bottom
        self.rep_start_time = None
        
        # Rep data collection
        self.current_rep_data = {
            'max_bar_tilt': 0.0,
            'max_elbow_flare': 0.0,
            'min_elbow_angle': 180.0,
            'max_elbow_angle': 0.0,
            'errors': []
        }
        
        print(" Rep Counter initialized with MID-RANGE START detection + INJURY RISK grading")
        print(f"   Bottom (flexed): < {self.FLEXED_THRESHOLD} degrees")
        print(f"   Top (extended): > {self.EXTENDED_THRESHOLD} degrees")
        print(f"    Can start counting from ANY position!")
    
    def _update_phase(self, avg_elbow_angle: float):
        """
        Update movement phase based on elbow angle
        
        Args:
            avg_elbow_angle: Average elbow angle from both arms
        """
        self.previous_phase = self.current_phase
        
        # Add to history
        self.elbow_angle_history.append(avg_elbow_angle)
        
        if len(self.elbow_angle_history) < 3:
            self.current_phase = "UNKNOWN"
            return
        
        # Get recent angles to determine direction
        recent_angles = list(self.elbow_angle_history)[-3:]
        angle_change = recent_angles[-1] - recent_angles[0]
        
        # Determine phase based on angle and trend
        if avg_elbow_angle > self.EXTENDED_THRESHOLD:
            # Arms extended - top position
            self.current_phase = "TOP"
        
        elif avg_elbow_angle < self.FLEXED_THRESHOLD:
            # Arms flexed - bottom position
            self.current_phase = "BOTTOM"
            # Mark that we reached bottom
            if self.rep_in_progress:
                self.reached_bottom = True
        
        elif angle_change < -self.MOVEMENT_THRESHOLD:
            # Angle decreasing - moving down (flexing)
            self.current_phase = "MOVING_DOWN"
        
        elif angle_change > self.MOVEMENT_THRESHOLD:
            # Angle increasing - moving up (extending)
            self.current_phase = "MOVING_UP"
        
        else:
            # In between, maintain last known phase
            pass
    
    def _detect_rep_completion(self) -> bool:
        """
        Detect if a rep has been completed
        NEW LOGIC: Start from any position, count when returning to flexed position after extension
        
        Returns:
            True if rep completed, False otherwise
        """
        # Start tracking a rep when we detect downward movement
        if not self.rep_in_progress:
            if self.current_phase == "MOVING_DOWN":
                # Start tracking a new rep
                self.rep_in_progress = True
                self.reached_bottom = False
                self.rep_start_time = time.time()
                return False
        
        # If rep is in progress, check for completion
        if self.rep_in_progress:
            # Rep completes when:
            # 1. We reached bottom (flexed position)
            # 2. We're now moving back up OR reached top
            if self.reached_bottom:
                if self.current_phase in ["MOVING_UP", "TOP"]:
                    # Rep complete! We went down and came back up
                    self.rep_in_progress = False
                    self.reached_bottom = False
                    return True
        
        return False
    
    def _collect_rep_data(self, features: Dict, errors: List):
        """
        Collect data during the rep
        
        Args:
            features: Current frame features
            errors: Current frame errors
        """
        # Track maximum bar tilt during rep
        bar_tilt = features.get('bar_vertical_tilt', 0)
        if bar_tilt > self.current_rep_data['max_bar_tilt']:
            self.current_rep_data['max_bar_tilt'] = bar_tilt
        
        # Track maximum elbow flare during rep
        avg_elbow = features.get('avg_elbow_angle', 0)
        if avg_elbow > self.current_rep_data['max_elbow_flare']:
            self.current_rep_data['max_elbow_flare'] = avg_elbow
        
        # Track elbow angle range
        if avg_elbow < self.current_rep_data['min_elbow_angle']:
            self.current_rep_data['min_elbow_angle'] = avg_elbow
        if avg_elbow > self.current_rep_data['max_elbow_angle']:
            self.current_rep_data['max_elbow_angle'] = avg_elbow
        
        # Collect errors (avoid duplicates)
        for error in errors:
            if error.error_type not in [e.error_type for e in self.current_rep_data['errors']]:
                self.current_rep_data['errors'].append(error)
    
    def _reset_rep_data(self):
        """Reset data collection for next rep"""
        self.current_rep_data = {
            'max_bar_tilt': 0.0,
            'max_elbow_flare': 0.0,
            'min_elbow_angle': 180.0,
            'max_elbow_angle': 0.0,
            'errors': []
        }
        self.rep_start_time = None
    
    def update(self, features: Dict, prediction: int, confidence: float, 
               errors: List) -> tuple[bool, Optional[RepAnalysis]]:
        """
        Update rep counter with new frame data
        
        Args:
            features: Extracted biomechanical features
            prediction: ML prediction (0=incorrect, 1=correct)
            confidence: Prediction confidence
            errors: List of detected errors
        
        Returns:
            Tuple of (rep_completed: bool, rep_analysis: Optional[RepAnalysis])
        """
        # Get average elbow angle
        avg_elbow_angle = features.get('avg_elbow_angle', 90)
        
        # Update phase
        self._update_phase(avg_elbow_angle)
        
        # Collect data during rep
        if self.rep_in_progress:
            self._collect_rep_data(features, errors)
        
        # Check if rep completed
        rep_completed = self._detect_rep_completion()
        
        if rep_completed:
            self.rep_count += 1
            
            # Calculate rep duration
            rep_duration = time.time() - self.rep_start_time if self.rep_start_time else 0
            
            # Create rep analysis
            # Inject bottom-position elbow angle into features for accurate grading
            enriched_features = features.copy()
            enriched_features['elbow_angle_min'] = self.current_rep_data['min_elbow_angle']
            enriched_features['elbow_angle_max'] = self.current_rep_data['max_elbow_angle']
            enriched_features['bar_tilt_max']    = self.current_rep_data['max_bar_tilt']

            rep_analysis = RepAnalysis(
                rep_number=self.rep_count,
                timestamp=time.time(),
                prediction=prediction,
                confidence=confidence,
                errors=self.current_rep_data['errors'].copy(),
                features=enriched_features,
                symmetry_grade=features.get('overall_symmetry_grade', 'C'),
                symmetry_score=features.get('symmetry_score', 0),
                bar_tilt_max=self.current_rep_data['max_bar_tilt'],
                elbow_flare_max=self.current_rep_data['max_elbow_flare'],
                elbow_angle_min=self.current_rep_data['min_elbow_angle'],
                elbow_angle_max=self.current_rep_data['max_elbow_angle'],
                rep_duration=rep_duration
            )
            
            # Store in history
            self.rep_history.append(rep_analysis)
            
            # Reset for next rep
            self._reset_rep_data()
            
            return True, rep_analysis
        
        return False, None
    
    def get_last_rep(self) -> Optional[RepAnalysis]:
        """Get the most recent rep analysis"""
        if self.rep_history:
            return self.rep_history[-1]
        return None
    
    def get_rep_history(self, n: int = 5) -> List[RepAnalysis]:
        """Get last N reps"""
        return list(self.rep_history)[-n:]
    
    def get_set_summary(self) -> Dict:
        """
        Get summary statistics for the entire set
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.rep_history:
            return {
                'total_reps': 0,
                'correct_reps': 0,
                'accuracy_rate': 0,
                'average_confidence': 0,
                'average_symmetry_score': 0,
                'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
            }
        
        total_reps = len(self.rep_history)
        correct_reps = sum(1 for rep in self.rep_history if rep.prediction == 1)
        avg_confidence = sum(rep.confidence for rep in self.rep_history) / total_reps
        avg_symmetry = sum(rep.symmetry_score for rep in self.rep_history) / total_reps
        
        # Grade distribution
        grade_dist = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for rep in self.rep_history:
            grade = rep.get_quality_grade()
            grade_dist[grade] += 1
        
        return {
            'total_reps': total_reps,
            'correct_reps': correct_reps,
            'accuracy_rate': (correct_reps / total_reps) * 100,
            'average_confidence': avg_confidence,
            'average_symmetry_score': avg_symmetry,
            'grade_distribution': grade_dist
        }
    
    def generate_set_report(self) -> str:
        """
        Generate detailed text report for the set
        NOW INCLUDES INJURY RISK SCORES!
        
        Returns:
            Formatted text report
        """
        if not self.rep_history:
            return "No reps completed yet."
        
        summary = self.get_set_summary()
        
        report = []
        report.append("=" * 70)
        report.append("SET SUMMARY REPORT")
        report.append("=" * 70)
        report.append("")
        
        report.append(f"Total Reps: {summary['total_reps']}")
        report.append(f"Correct Technique: {summary['correct_reps']} ({summary['accuracy_rate']:.1f}%)")
        report.append(f"Average Confidence: {summary['average_confidence']*100:.1f}%")
        report.append(f"Average Symmetry Score: {summary['average_symmetry_score']:.3f}")
        report.append("")
        
        report.append("Grade Distribution:")
        for grade, count in summary['grade_distribution'].items():
            if count > 0:
                report.append(f"  {grade}: {count} reps")
        report.append("")
        
        # NEW: Average injury risk
        avg_risk = sum(rep.get_risk_score() for rep in self.rep_history) / len(self.rep_history)
        report.append(f"AVERAGE INJURY RISK: {avg_risk:.1f}/100")
        if avg_risk >= 70:
            report.append("    CRITICAL - Immediate form correction needed!")
        elif avg_risk >= 50:
            report.append("    HIGH - Significant injury risk present")
        elif avg_risk >= 30:
            report.append("    MODERATE - Attention to form recommended")
        elif avg_risk >= 15:
            report.append("  OK MINOR - Small improvements possible")
        else:
            report.append("  OK EXCELLENT - Safe form maintained!")
        report.append("")
        
        report.append("=" * 70)
        report.append("REP-BY-REP BREAKDOWN")
        report.append("=" * 70)
        report.append("")
        
        for rep in self.rep_history:
            grade = rep.get_quality_grade()
            risk = rep.get_risk_score()
            
            report.append(f"REP #{rep.rep_number}")
            report.append(f"  Grade: {grade} | Risk: {risk:.0f}/100")
            report.append(f"  Technique: {'CORRECT' if rep.prediction == 1 else 'INCORRECT'}")
            report.append(f"  Confidence: {rep.confidence*100:.1f}%")
            report.append(f"  Symmetry: {rep.symmetry_grade} ({rep.symmetry_score:.3f})")
            report.append(f"  Max Bar Tilt: {rep.bar_tilt_max*100:.1f}%")
            report.append(f"  Max Elbow Flare: {rep.elbow_flare_max:.1f} degrees")
            report.append(f"  Elbow Range: {rep.elbow_angle_min:.1f} degrees - {rep.elbow_angle_max:.1f} degrees")
            report.append(f"  Duration: {rep.rep_duration:.1f}s")
            
            # NEW: Show grade explanation for poor grades
            if grade in ['C', 'D', 'F']:
                explanation = rep.get_grade_explanation()
                if explanation:
                    report.append(f"  Issues: {', '.join(explanation)}")
            
            if rep.errors:
                report.append(f"  Errors: {len(rep.errors)}")
                for error in rep.errors:
                    report.append(f"    - {error.error_type} ({error.severity})")
            else:
                report.append(f"  Errors: None")
            
            report.append("")
        
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        return "\n".join(report)