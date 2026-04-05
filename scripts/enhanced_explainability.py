import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

class MuscleGroup(Enum):
    """Major muscle groups involved in bench press and shoulder press"""
    ANTERIOR_DELTOID = "Anterior Deltoid"
    PECTORALIS_MAJOR = "Pectoralis Major"
    ROTATOR_CUFF = "Rotator Cuff (SITS muscles)"
    TRICEPS = "Triceps Brachii"
    BICEPS = "Biceps Brachii"
    SERRATUS_ANTERIOR = "Serratus Anterior"
    TRAPEZIUS = "Trapezius"
    RHOMBOIDS = "Rhomboids"
    LATISSIMUS_DORSI = "Latissimus Dorsi"

@dataclass
class AffectedMuscle:
    """Represents a muscle affected by poor technique"""
    name: str
    function: str
    location: str
    risk_level: str  # "HIGH", "MEDIUM", "LOW"

@dataclass
class BiomechanicalError:
    """Represents a detected biomechanical error with medical context"""
    error_type: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    description: str
    affected_muscles: List[AffectedMuscle]
    injury_risks: List[str]
    correction_steps: List[str]
    joints_at_risk: List[str]

class AnatomicalExpertSystem:
    """
    Expert system for biomechanical analysis with medical-grade feedback
    """
    
    def __init__(self):
        """Initialize the expert system with medical knowledge base"""
        self.error_rules = self._initialize_error_rules()
    
    def _initialize_error_rules(self) -> Dict[str, BiomechanicalError]:
        """
        Initialize comprehensive error detection rules with medical information
        """
        return {
            'extreme_elbow_flare': BiomechanicalError(
                error_type="Extreme Elbow Flare (>120 degrees)",
                severity="CRITICAL",
                description="Elbows are positioned far beyond safe range, creating excessive stress on anterior shoulder structures",
                affected_muscles=[
                    AffectedMuscle(
                        name="Anterior Deltoid",
                        function="Shoulder flexion and horizontal adduction",
                        location="Front of shoulder",
                        risk_level="HIGH"
                    ),
                    AffectedMuscle(
                        name="Rotator Cuff (SITS muscles)",
                        function="Stabilize shoulder joint, prevent humeral head migration",
                        location="Deep shoulder muscles (Supraspinatus, Infraspinatus, Teres Minor, Subscapularis)",
                        risk_level="CRITICAL"
                    ),
                    AffectedMuscle(
                        name="Pectoralis Major",
                        function="Primary mover for horizontal shoulder adduction (pressing)",
                        location="Chest - connects sternum/clavicle to humerus",
                        risk_level="HIGH"
                    )
                ],
                injury_risks=[
                    "Rotator cuff impingement syndrome (subacromial space compression)",
                    "Anterior shoulder instability and potential dislocation",
                    "Biceps tendinitis due to excessive strain on long head of biceps",
                    "Labral tears (glenoid labrum damage)",
                    "Chronic shoulder pain and inflammation",
                    "Pectoralis major strain or partial tear at musculotendinous junction"
                ],
                correction_steps=[
                    "STOP IMMEDIATELY - Risk of acute injury is high",
                    "Reduce weight by 40-50% for next session",
                    "Keep elbows at 45-75 degrees from torso (imagine squeezing oranges in armpits)",
                    "Focus on tucking elbows slightly toward ribs during descent",
                    "Consider working with a certified strength coach",
                    "If pain persists, consult sports medicine physician or physical therapist"
                ],
                joints_at_risk=["Shoulder (glenohumeral joint)", "Acromioclavicular (AC) joint", "Scapulothoracic joint"]
            ),
            
            'excessive_elbow_flare': BiomechanicalError(
                error_type="Excessive Elbow Flare (>75 degrees)",
                severity="HIGH",
                description="Elbows are too far from body, creating excessive stress on anterior shoulder structures",
                affected_muscles=[
                    AffectedMuscle(
                        name="Anterior Deltoid",
                        function="Shoulder flexion and horizontal adduction",
                        location="Front of shoulder",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Rotator Cuff (SITS muscles)",
                        function="Stabilize shoulder joint, prevent humeral head migration",
                        location="Deep shoulder muscles (Supraspinatus, Infraspinatus, Teres Minor, Subscapularis)",
                        risk_level="HIGH"
                    ),
                    AffectedMuscle(
                        name="Pectoralis Major",
                        function="Primary mover for horizontal shoulder adduction (pressing)",
                        location="Chest - connects sternum/clavicle to humerus",
                        risk_level="MEDIUM"
                    )
                ],
                injury_risks=[
                    "Rotator cuff tendinopathy (chronic inflammation)",
                    "Subacromial impingement leading to bursitis",
                    "Anterior shoulder capsule strain",
                    "Increased risk of shoulder instability over time",
                    "AC joint stress and potential degeneration"
                ],
                correction_steps=[
                    "Reduce weight by 20-30% to practice proper form",
                    "Tuck elbows to 45-75 degrees angle from torso",
                    "Think 'bend the bar' cue - externally rotate shoulders slightly",
                    "Strengthen rotator cuff with band external rotations",
                    "Perform face pulls to improve scapular stability",
                    "Film yourself from above to monitor elbow position"
                ],
                joints_at_risk=["Shoulder (glenohumeral joint)", "Acromioclavicular (AC) joint"]
            ),
            
            'insufficient_scapular_retraction': BiomechanicalError(
                error_type="Insufficient Scapular Retraction",
                severity="MEDIUM",
                description="Shoulder blades are not properly pulled together and down, reducing stability and power transfer",
                affected_muscles=[
                    AffectedMuscle(
                        name="Rhomboids",
                        function="Retract and stabilize scapula",
                        location="Upper back between spine and shoulder blades",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Trapezius (middle/lower fibers)",
                        function="Scapular retraction and depression",
                        location="Mid to lower back, connecting spine to scapula",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Serratus Anterior",
                        function="Scapular protraction and upward rotation",
                        location="Side of ribs, under armpit",
                        risk_level="LOW"
                    ),
                    AffectedMuscle(
                        name="Rotator Cuff",
                        function="Dynamic shoulder stabilization",
                        location="Deep shoulder muscles",
                        risk_level="MEDIUM"
                    )
                ],
                injury_risks=[
                    "Scapular winging and dyskinesis",
                    "Reduced force production (10-20% power loss)",
                    "Anterior shoulder instability",
                    "Increased stress on rotator cuff during press",
                    "Upper trap dominance and neck tension",
                    "Thoracic outlet syndrome symptoms"
                ],
                correction_steps=[
                    "Before unracking bar: Pull shoulder blades together and DOWN",
                    "Imagine squeezing a pencil between shoulder blades",
                    "Create a 'shelf' with your upper back on the bench",
                    "Maintain retraction throughout entire movement",
                    "Strengthen mid/lower traps: prone Y-raises, face pulls",
                    "Improve thoracic spine mobility with foam rolling",
                    "Practice scapular wall slides daily"
                ],
                joints_at_risk=["Scapulothoracic joint", "Shoulder (glenohumeral joint)"]
            ),
            
            'uneven_bar_path': BiomechanicalError(
                error_type="Uneven Bar Path (Tilted Bar)",
                severity="HIGH",
                description="Bar is not level during movement, indicating bilateral strength imbalance or grip issues",
                affected_muscles=[
                    AffectedMuscle(
                        name="Pectoralis Major (unilateral dominance)",
                        function="Primary pressing muscle",
                        location="Chest - one side working harder",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Anterior Deltoid (unilateral)",
                        function="Shoulder flexion and pressing",
                        location="Front of shoulder - asymmetric loading",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Triceps (unilateral)",
                        function="Elbow extension",
                        location="Back of arm - one side overworking",
                        risk_level="LOW"
                    )
                ],
                injury_risks=[
                    "One side handling 60-70% of load (dangerous imbalance)",
                    "Increased risk of bar dropping toward weaker side",
                    "Unilateral muscle strain on dominant side",
                    "Spinal rotation and torque through thoracic spine",
                    "Development of chronic postural asymmetries",
                    "Risk of acute muscle tear if bar slips"
                ],
                correction_steps=[
                    "Check grip: Hands must be equidistant from center of bar",
                    "Use bar markings or tape to ensure even hand placement",
                    "Reduce weight by 25-30% to practice balanced pressing",
                    "Strengthen weaker side: Single-arm dumbbell press",
                    "Check for underlying flexibility imbalances",
                    "Ensure bench is level and you're centered on it",
                    "Have spotter verify bar stays level throughout set"
                ],
                joints_at_risk=["Both shoulders", "Wrists", "Thoracic spine"]
            ),
            
            'bar_off_center': BiomechanicalError(
                error_type="Bar Path Off-Center",
                severity="MEDIUM",
                description="Bar is not tracking over the correct path relative to body midline",
                affected_muscles=[
                    AffectedMuscle(
                        name="Pectoralis Major",
                        function="Horizontal pressing",
                        location="Chest - suboptimal fiber engagement",
                        risk_level="LOW"
                    ),
                    AffectedMuscle(
                        name="Core Stabilizers",
                        function="Maintain spinal alignment",
                        location="Abdominals and obliques",
                        risk_level="MEDIUM"
                    )
                ],
                injury_risks=[
                    "Reduced force production efficiency",
                    "Increased stress on one shoulder",
                    "Potential for bar to drift further off-path",
                    "Core instability and compensatory patterns",
                    "Uneven muscle development over time"
                ],
                correction_steps=[
                    "Ensure you're centered on the bench",
                    "Bar should track over mid-chest to lower sternum",
                    "Check foot placement - both feet flat and stable",
                    "Strengthen core with planks and dead bugs",
                    "Practice bar path with lighter weight",
                    "Use camera feedback to assess alignment"
                ],
                joints_at_risk=["Shoulders", "Thoracic spine"]
            ),
            
            'shoulder_asymmetry': BiomechanicalError(
                error_type="Shoulder Height Asymmetry",
                severity="HIGH",
                description="Left and right shoulders are at different heights, indicating bilateral imbalance or postural issues",
                affected_muscles=[
                    AffectedMuscle(
                        name="Trapezius (unilateral elevation)",
                        function="Shoulder elevation and scapular control",
                        location="One side elevated higher",
                        risk_level="HIGH"
                    ),
                    AffectedMuscle(
                        name="Levator Scapulae",
                        function="Elevates scapula",
                        location="Neck to shoulder blade",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Rotator Cuff (asymmetric loading)",
                        function="Shoulder stabilization",
                        location="Deep shoulder muscles - uneven stress",
                        risk_level="HIGH"
                    )
                ],
                injury_risks=[
                    "Uneven loading pattern (60-40% weight distribution)",
                    "One shoulder handling excessive load",
                    "Risk of unilateral rotator cuff injury",
                    "Cervical spine rotation and neck strain",
                    "Development of permanent postural asymmetry",
                    "Increased risk of labral tear on elevated side",
                    "Potential thoracic outlet syndrome"
                ],
                correction_steps=[
                    "Check grip width - must be equal on both sides",
                    "Ensure bench setup is level",
                    "Reduce weight by 20-30%",
                    "Perform unilateral exercises: single-arm dumbbell press",
                    "Address postural imbalances: check for scoliosis",
                    "Strengthen weaker side specifically",
                    "Consider physical therapy evaluation if persistent",
                    "Practice shoulder packing exercises bilaterally"
                ],
                joints_at_risk=["Both shoulders (glenohumeral joints)", "Cervical spine", "Thoracic spine"]
            ),
            
            'elbow_asymmetry': BiomechanicalError(
                error_type="Elbow Height/Angle Asymmetry",
                severity="HIGH",
                description="Left and right elbows moving at different heights or angles, indicating strength imbalance",
                affected_muscles=[
                    AffectedMuscle(
                        name="Triceps (unilateral dominance)",
                        function="Elbow extension - one side stronger",
                        location="Back of arm",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Pectoralis Major (unilateral)",
                        function="Pressing - asymmetric contribution",
                        location="Chest",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Anterior Deltoid (unilateral)",
                        function="Shoulder pressing - imbalanced",
                        location="Front shoulder",
                        risk_level="MEDIUM"
                    )
                ],
                injury_risks=[
                    "One elbow handling 10-30% more load than other",
                    "Unilateral elbow tendinitis risk",
                    "Bar rotation during pressing motion",
                    "Compensatory movement patterns developing",
                    "Chronic pain in dominant elbow over time",
                    "Risk of acute strain if imbalance worsens"
                ],
                correction_steps=[
                    "Reduce weight by 25%",
                    "Reset starting position - ensure elbows level",
                    "Strengthen weaker arm: single-arm cable press",
                    "Address flexibility differences between sides",
                    "Check for previous injuries affecting range of motion",
                    "Practice tempo training (3-1-3 tempo) for control",
                    "Use mirrors or camera to monitor elbow symmetry",
                    "Consider unilateral dumbbell work to identify imbalance"
                ],
                joints_at_risk=["Elbows", "Shoulders", "Wrists"]
            ),
            
            'wrist_asymmetry': BiomechanicalError(
                error_type="Wrist Height Asymmetry",
                severity="MEDIUM",
                description="Wrists at different heights, indicating grip issues or bar tilt",
                affected_muscles=[
                    AffectedMuscle(
                        name="Wrist Flexors/Extensors",
                        function="Grip and wrist stabilization",
                        location="Forearms - asymmetric strain",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Forearm Muscles",
                        function="Grip strength and control",
                        location="Forearms",
                        risk_level="LOW"
                    )
                ],
                injury_risks=[
                    "Bar is tilted - high risk of dropping toward lower side",
                    "Uneven wrist strain and potential sprain",
                    "Grip strength imbalances",
                    "Risk of bar slipping during heavy sets",
                    "Chronic wrist pain on one side",
                    "Development of grip-related compensation patterns"
                ],
                correction_steps=[
                    "Check grip: hands equidistant from bar center",
                    "Use bar markings or measure grip width",
                    "Strengthen grip bilaterally: farmer's carries, dead hangs",
                    "Practice wrist stability exercises",
                    "Ensure wrists are neutral (not bent back excessively)",
                    "Consider wrist wraps for support during learning phase",
                    "Address any previous wrist injuries"
                ],
                joints_at_risk=["Wrists", "Elbows"]
            ),
            
            'grip_too_wide': BiomechanicalError(
                error_type="Grip Too Wide",
                severity="HIGH",
                description="Forearms angled outward at bottom position — grip is too wide, wrists not stacked over elbows",
                affected_muscles=[
                    AffectedMuscle(
                        name="Rotator Cuff",
                        function="Shoulder stabilization",
                        location="Shoulder - excessive abduction stress",
                        risk_level="HIGH"
                    ),
                    AffectedMuscle(
                        name="Anterior Deltoid",
                        function="Primary pressing muscle",
                        location="Front of shoulder",
                        risk_level="MEDIUM"
                    )
                ],
                injury_risks=[
                    "Excessive shoulder abduction causing rotator cuff impingement",
                    "AC joint stress from wide pressing angle",
                    "Reduced pressing power and stability",
                    "Long term rotator cuff tears"
                ],
                correction_steps=[
                    "Bring hands closer so forearms are vertical at bottom position",
                    "Wrists should be directly above elbows at starting position",
                    "Grip just outside shoulder width is optimal",
                    "Forearms should be perpendicular to floor at bottom"
                ],
                joints_at_risk=["Shoulder", "AC Joint", "Elbow"]
            ),

            'grip_too_narrow': BiomechanicalError(
                error_type="Grip Too Narrow",
                severity="MEDIUM",
                description="Forearms angled inward at bottom position — grip is too narrow, causing elbow and wrist stress",
                affected_muscles=[
                    AffectedMuscle(
                        name="Triceps",
                        function="Elbow extension",
                        location="Back of upper arm - overloaded",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Wrist Flexors",
                        function="Wrist stabilization",
                        location="Forearm - inward strain",
                        risk_level="MEDIUM"
                    )
                ],
                injury_risks=[
                    "Excessive elbow stress from narrow pressing angle",
                    "Wrist strain from inward forearm angle",
                    "Reduced deltoid activation",
                    "Elbow tendinitis over time"
                ],
                correction_steps=[
                    "Widen grip so forearms are vertical at bottom position",
                    "Wrists should be directly above elbows",
                    "Grip just outside shoulder width is optimal",
                    "Check forearm angle — aim for straight up and down"
                ],
                joints_at_risk=["Elbow", "Wrist"]
            ),

            'hip_tilt': BiomechanicalError(
                error_type="Hip Tilt (Pelvic Asymmetry)",
                severity="MEDIUM",
                description="Hips are not level, indicating leg length discrepancy, core weakness, or bench positioning issues",
                affected_muscles=[
                    AffectedMuscle(
                        name="Quadratus Lumborum",
                        function="Lateral spine flexion and hip hiking",
                        location="Lower back - one side overactive",
                        risk_level="MEDIUM"
                    ),
                    AffectedMuscle(
                        name="Gluteus Medius",
                        function="Hip stabilization",
                        location="Side of hip",
                        risk_level="LOW"
                    ),
                    AffectedMuscle(
                        name="Obliques",
                        function="Core stability and rotation control",
                        location="Sides of abdomen",
                        risk_level="MEDIUM"
                    )
                ],
                injury_risks=[
                    "May indicate leg length discrepancy",
                    "Core instability during pressing",
                    "Increased stress on lower back",
                    "Sacroiliac (SI) joint irritation",
                    "Asymmetric force transfer from legs through core",
                    "Long-term risk of chronic lower back pain"
                ],
                correction_steps=[
                    "Check foot placement - both feet flat on floor",
                    "Ensure even weight distribution through both legs",
                    "Strengthen core: planks, dead bugs, pallof press",
                    "Assess for leg length discrepancy (may need PT evaluation)",
                    "Work on hip mobility and flexibility",
                    "Practice proper bench setup and leg drive technique",
                    "Consider core strengthening program before increasing weight"
                ],
                joints_at_risk=["Sacroiliac (SI) joint", "Lumbar spine", "Hips"]
            ),
            
            'overall_poor_symmetry': BiomechanicalError(
                error_type="Overall Poor Bilateral Symmetry",
                severity="HIGH",
                description="Multiple joints showing asymmetry, indicating systemic strength imbalance or movement dysfunction",
                affected_muscles=[
                    AffectedMuscle(
                        name="Multiple muscle groups bilaterally",
                        function="Systemic imbalance across pressing chain",
                        location="Entire upper body kinetic chain",
                        risk_level="HIGH"
                    )
                ],
                injury_risks=[
                    "High risk of injury due to multiple imbalances",
                    "Compensation patterns throughout kinetic chain",
                    "Uneven stress distribution across all joints",
                    "Risk of acute injury if weight is too heavy",
                    "Chronic pain development likely without correction",
                    "Poor motor control and movement quality"
                ],
                correction_steps=[
                    "STOP current training program",
                    "Reduce weight by 40-50%",
                    "Seek evaluation from certified strength coach or physical therapist",
                    "Begin corrective exercise program (4-8 weeks)",
                    "Focus on unilateral exercises to identify specific weaknesses",
                    "Address flexibility and mobility imbalances",
                    "Relearn movement patterns with very light weight",
                    "Consider functional movement screen (FMS) assessment"
                ],
                joints_at_risk=["All upper body joints", "Spine"]
            ),
        }
    
    def analyze_technique(self, features: Dict, exercise_config=None) -> List[BiomechanicalError]:
        """
        Analyze biomechanical features with exercise-specific rules
        
        Args:
            features: Dictionary of extracted biomechanical features
            exercise_config: Exercise configuration dictionary (optional)
        
        Returns:
            List of detected BiomechanicalError objects
        """
        detected_errors = []
        
        # Get exercise-specific checks
        if exercise_config:
            checks = exercise_config['specific_checks']
        else:
            # Default: check everything
            checks = {
                'check_elbow_flare': True,
                'check_bar_path': True,
                'check_symmetry': True,
                'check_scapular_retraction': True,
                'check_hip_tilt': True,
            }
        
        # Rule 1: Check elbow angle (always checked for both exercises)
        if checks.get('check_elbow_flare', True):
            avg_elbow = features.get('avg_elbow_angle', 0)
            if avg_elbow > 120:
                detected_errors.append(self.error_rules['extreme_elbow_flare'])
            elif avg_elbow > 75:
                detected_errors.append(self.error_rules['excessive_elbow_flare'])
        
        # Rule 2: Check scapular retraction (only if hips visible for accurate measurement)
        if checks.get('check_scapular_retraction', True):
            retraction_normalized = features.get('retraction_normalized', 0)
            if retraction_normalized > 0.15:
                detected_errors.append(self.error_rules['insufficient_scapular_retraction'])
        
        # Rule 3: Check bar path (always important)
        if checks.get('check_bar_path', True):
            bar_vertical_tilt = features.get('bar_vertical_tilt', 0)
            if bar_vertical_tilt > 0.05:
                detected_errors.append(self.error_rules['uneven_bar_path'])
            
            bar_center_offset = features.get('bar_center_offset', 0)
            if bar_center_offset > 0.15:
                detected_errors.append(self.error_rules['bar_off_center'])
        
        # Rule 4: Check bilateral symmetry (always important)
        if checks.get('check_symmetry', True):
            # Shoulder symmetry
            shoulder_height_diff = features.get('shoulder_height_diff', 0)
            shoulder_angle_diff = features.get('shoulder_angle_diff', 0)
            if shoulder_height_diff > 0.05 or shoulder_angle_diff > 15:
                detected_errors.append(self.error_rules['shoulder_asymmetry'])
            
            # Elbow symmetry
            elbow_height_diff = features.get('elbow_height_diff', 0)
            elbow_angle_diff = features.get('elbow_angle_diff', 0)
            if elbow_height_diff > 0.05 or elbow_angle_diff > 15:
                detected_errors.append(self.error_rules['elbow_asymmetry'])
            
            # Wrist symmetry
            wrist_height_diff = features.get('wrist_height_diff', 0)
            if wrist_height_diff > 0.05:
                detected_errors.append(self.error_rules['wrist_asymmetry'])
        
        # Rule 5: Check hip tilt (only if hips are visible - shoulder press)
        if checks.get('check_hip_tilt', True) and features.get('has_hip_data', False):
            hip_height_diff = features.get('hip_height_diff', 0)
            if hip_height_diff > 0.05:
                detected_errors.append(self.error_rules['hip_tilt'])
        # Rule 6: Forearm verticality check (grip width) - ONLY at bottom position
        # At bottom of shoulder press, forearms should be vertical
        # Measure angle of forearm line from vertical using wrist-elbow coordinates
        if checks.get('check_elbow_flare', True) and features.get('has_hip_data', False):
            import math
            # Only check at bottom position (elbow angle 60-110 degrees)
            avg_elbow = features.get('avg_elbow_angle', 90)
            if 55 <= avg_elbow <= 110:
                # Left forearm angle from vertical
                lw_x = features.get('left_wrist_x', 0)
                lw_y = features.get('left_wrist_y', 0)
                le_x = features.get('left_elbow_x', 0)
                le_y = features.get('left_elbow_y', 0)

                # Right forearm angle from vertical
                rw_x = features.get('right_wrist_x', 0)
                rw_y = features.get('right_wrist_y', 0)
                re_x = features.get('right_elbow_x', 0)
                re_y = features.get('right_elbow_y', 0)

                # Calculate horizontal deviation of forearm from vertical
                # dy should be much larger than dx for vertical forearm
                left_dy  = abs(lw_y - le_y) + 1e-6
                left_dx  = lw_x - le_x   # positive = wrist right of elbow
                right_dy = abs(rw_y - re_y) + 1e-6
                right_dx = rw_x - re_x   # positive = wrist right of elbow

                # Angle from vertical in degrees
                left_angle  = math.degrees(math.atan(abs(left_dx)  / left_dy))
                right_angle = math.degrees(math.atan(abs(right_dx) / right_dy))
                avg_forearm_angle = (left_angle + right_angle) / 2

                # Too wide: wrist is further out than elbow (forearm angled outward)
                left_too_wide  = left_dx  < -0.02   # wrist left of elbow (left arm flaring)
                right_too_wide = right_dx >  0.02   # wrist right of elbow (right arm flaring)

                # Too narrow: wrist is further in than elbow
                left_too_narrow  = left_dx  >  0.05
                right_too_narrow = right_dx < -0.05

                if (left_too_wide or right_too_wide) and avg_forearm_angle > 20:
                    detected_errors.append(self.error_rules['grip_too_wide'])
                elif (left_too_narrow or right_too_narrow) and avg_forearm_angle > 20:
                    detected_errors.append(self.error_rules['grip_too_narrow'])

        
        # Rule 6: Check overall symmetry
        symmetry_score = features.get('symmetry_score', 0)
        if symmetry_score > 0.15:
            detected_errors.append(self.error_rules['overall_poor_symmetry'])
        
        return detected_errors
    
    def generate_detailed_report(self, errors: List[BiomechanicalError]) -> str:
        """
        Generate comprehensive medical report from detected errors
        
        Args:
            errors: List of detected biomechanical errors
        
        Returns:
            Formatted text report
        """
        if not errors:
            return " NO ERRORS DETECTED\n\nTechnique appears correct. Continue maintaining good form."
        
        report = []
        report.append("=" * 70)
        report.append("BIOMECHANICAL ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        report.append(f"Total Errors Detected: {len(errors)}")
        report.append(f"Highest Severity: {errors[0].severity}")
        report.append("")
        
        for i, error in enumerate(errors, 1):
            report.append("=" * 70)
            report.append(f"ERROR #{i}: {error.error_type}")
            report.append(f"SEVERITY: {error.severity}")
            report.append("=" * 70)
            report.append("")
            
            report.append(error.description)
            report.append("")
            
            report.append("AFFECTED ANATOMY:")
            for muscle in error.affected_muscles:
                report.append(f"  - {muscle.name}")
                report.append(f"    Function: {muscle.function}")
                report.append(f"    Location: {muscle.location}")
                report.append(f"    Risk Level: {muscle.risk_level}")
                report.append("")
            
            report.append("JOINTS AT RISK:")
            for joint in error.joints_at_risk:
                report.append(f"  - {joint}")
            report.append("")
            
            report.append("INJURY RISK FACTORS:")
            for risk in error.injury_risks:
                report.append(f"  - {risk}")
            report.append("")
            
            report.append("CORRECTION PROTOCOL:")
            for i, step in enumerate(error.correction_steps, 1):
                report.append(f"  {i}. {step}")
            report.append("")
        
        report.append("=" * 70)
        report.append("RECOMMENDATIONS")
        report.append("=" * 70)
        report.append("")
        
        if any(e.severity == "CRITICAL" for e in errors):
            report.append("  CRITICAL ISSUES DETECTED")
            report.append("  - Stop current set immediately")
            report.append("  - Reduce weight by 40-50%")
            report.append("  - Focus on technique correction before adding weight")
            report.append("  - Consider consulting with a certified professional")
        elif any(e.severity == "HIGH" for e in errors):
            report.append("  HIGH-RISK ISSUES DETECTED")
            report.append("  - Reduce weight by 20-30%")
            report.append("  - Address form issues before progressing")
            report.append("  - Film your sets to monitor improvement")
        else:
            report.append("ℹ  MODERATE ISSUES DETECTED")
            report.append("  - Make corrections with current weight")
            report.append("  - Focus on quality movement patterns")
            report.append("  - Monitor for improvement over next few sessions")
        
        report.append("")
        report.append("=" * 70)
        report.append("IMPORTANT DISCLAIMER")
        report.append("=" * 70)
        report.append("")
        report.append("This analysis is based on computer vision and biomechanical")
        report.append("algorithms. For personalized medical advice, diagnosis, or")
        report.append("treatment of injuries, always consult with:")
        report.append("  - Licensed Physical Therapist")
        report.append("  - Sports Medicine Physician")
        report.append("  - Certified Strength and Conditioning Specialist (CSCS)")
        report.append("")
        report.append("If you experience pain, stop immediately and seek professional")
        report.append("medical evaluation.")
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)

def draw_anatomical_overlay(frame, landmarks, errors: List[BiomechanicalError]):
    """
    Draw anatomical highlights on frame showing affected areas
    
    Args:
        frame: OpenCV frame
        landmarks: MediaPipe pose landmarks
        errors: List of detected errors
    
    Returns:
        Frame with anatomical overlays
    """
    if not errors:
        return frame
    
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Get landmark positions
    lm = landmarks[0]
    left_shoulder = (int(lm[11].x * w), int(lm[11].y * h))
    right_shoulder = (int(lm[12].x * w), int(lm[12].y * h))
    left_elbow = (int(lm[13].x * w), int(lm[13].y * h))
    right_elbow = (int(lm[14].x * w), int(lm[14].y * h))
    left_wrist = (int(lm[15].x * w), int(lm[15].y * h))
    right_wrist = (int(lm[16].x * w), int(lm[16].y * h))
    
    # Determine severity color
    max_severity = errors[0].severity
    if max_severity == "CRITICAL":
        color = (0, 0, 255)  # Red
        alpha = 0.4
    elif max_severity == "HIGH":
        color = (0, 100, 255)  # Orange
        alpha = 0.3
    else:
        color = (0, 165, 255)  # Yellow
        alpha = 0.2
    
    # Draw affected areas based on error type
    for error in errors:
        if "elbow" in error.error_type.lower():
            # Highlight elbow region
            cv2.circle(overlay, left_elbow, 50, color, -1)
            cv2.circle(overlay, right_elbow, 50, color, -1)
        
        if "shoulder" in error.error_type.lower() or "scapular" in error.error_type.lower():
            # Highlight shoulder region
            cv2.circle(overlay, left_shoulder, 45, color, -1)
            cv2.circle(overlay, right_shoulder, 45, color, -1)
        
        if "wrist" in error.error_type.lower() or "bar" in error.error_type.lower():
            # Highlight wrist/bar region
            cv2.circle(overlay, left_wrist, 35, color, -1)
            cv2.circle(overlay, right_wrist, 35, color, -1)
    
    # Blend overlay
    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    return frame