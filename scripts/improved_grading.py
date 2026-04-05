import numpy as np

def calculate_injury_risk_score(features, errors):
    risk = 0.0

    avg_elbow = features.get('avg_elbow_angle', 75)
    if avg_elbow > 120:
        risk += 50
    elif avg_elbow > 90:
        risk += 30
    elif avg_elbow > 75:
        risk += 10

    if features.get('extreme_flare', 0) == 1:
        risk += 30

    elbow_diff = features.get('elbow_angle_diff', 0)
    if elbow_diff > 20:
        risk += 25
    elif elbow_diff > 15:
        risk += 15
    elif elbow_diff > 10:
        risk += 5

    alignment = features.get('alignment_score', 1.0)
    if alignment < 0.7:
        risk += 15
    elif alignment < 0.85:
        risk += 8

    bar_tilt = features.get('bar_vertical_tilt', 0)
    if bar_tilt > 0.10:
        risk += 20
    elif bar_tilt > 0.05:
        risk += 10

    shoulder_diff = features.get('shoulder_height_diff', 0)
    if shoulder_diff > 0.08:
        risk += 20
    elif shoulder_diff > 0.05:
        risk += 10

    for error in errors:
        if error.severity == "CRITICAL":
            risk += 15
        elif error.severity == "HIGH":
            risk += 10
        elif error.severity == "MEDIUM":
            risk += 5

    return min(risk, 100)

def apply_biomechanical_overrides(features, errors, prediction):
    """
    Override ML prediction with strict biomechanical rules.
    Risk score is checked FIRST — high risk always gives F/D regardless of other rules.
    """
    bottom_elbow  = features.get('elbow_angle_min', features.get('avg_elbow_angle', 75))
    bar_tilt      = features.get('bar_tilt_max', features.get('bar_vertical_tilt', 0))
    shoulder_diff = features.get('shoulder_height_diff', 0)
    elbow_diff    = features.get('elbow_angle_diff', 0)

    # RULE 1: Risk score first — always wins
    risk_score = calculate_injury_risk_score(features, errors)
    if risk_score >= 70:
        return 0, "F", f"Critical injury risk ({risk_score:.0f}/100) — Immediate form correction needed"
    if risk_score >= 50:
        return 0, "D", f"High injury risk ({risk_score:.0f}/100) — Significant form issues detected"

    # RULE 2: Any CRITICAL error => F
    for error in errors:
        if error.severity == "CRITICAL":
            return 0, "F", f"Critical error: {error.error_type}"

    # RULE 3: Elbow flare at bottom position
    if bottom_elbow > 120 and features.get('extreme_flare', 0) == 1:
        return 0, "F", f"Extreme Elbow Flare ({bottom_elbow:.0f} degrees) — Critical rotator cuff injury risk"

    # RULE 4: Uneven bar
    if bar_tilt > 0.10:
        return 0, "D", "Severely Uneven Bar Path — High risk of unilateral shoulder impingement"
    elif bar_tilt > 0.05:
        return 0, "C", "Uneven Bar Path — Asymmetric loading on shoulder joints"

    # RULE 5: Shoulder asymmetry
    if shoulder_diff > 0.08:
        return 0, "D", "Significant Shoulder Imbalance — Risk of rotator cuff overload"
    elif shoulder_diff > 0.05:
        return 0, "C", "Shoulder Height Asymmetry — Uneven muscle recruitment"

    # RULE 6: Elbow asymmetry
    if elbow_diff > 20:
        return 0, "D", "Severe Elbow Asymmetry — Unilateral joint stress risk"
    elif elbow_diff > 15:
        return 0, "C", "Elbow Angle Asymmetry — Uneven pressing mechanics"

    return prediction, None, None

def get_grade_from_risk_score(risk_score, prediction, confidence):
    if prediction == 1:
        if confidence > 0.9:
            return 'A'
        elif confidence > 0.7:
            return 'B'
        else:
            return 'C'

    if risk_score >= 70:
        return 'F'
    elif risk_score >= 50:
        return 'D'
    elif risk_score >= 30:
        return 'C'
    elif risk_score >= 15:
        return 'D'
    else:
        return 'C'

def get_improved_grade_with_explanation(features, prediction, confidence, errors):
    overridden_prediction, forced_grade, override_reason = apply_biomechanical_overrides(
        features, errors, prediction
    )

    risk_score = calculate_injury_risk_score(features, errors)

    if forced_grade:
        explanation = [override_reason] if override_reason else []
        return forced_grade, risk_score, explanation

    grade = get_grade_from_risk_score(risk_score, overridden_prediction, confidence)

    explanation = []
    avg_elbow = features.get('avg_elbow_angle', 75)

    if overridden_prediction == 1:
        if confidence > 0.9:
            explanation.append("Excellent form — minimal injury risk")
        else:
            explanation.append("Good form — minor issues detected")
    else:
        if avg_elbow > 90:
            explanation.append(f"Elbow angle {avg_elbow:.0f} degrees — too wide, risk of shoulder injury")
        if features.get('bar_vertical_tilt', 0) > 0.05:
            explanation.append("Uneven bar path — asymmetric shoulder loading")
        if features.get('shoulder_height_diff', 0) > 0.05:
            explanation.append("Shoulder imbalance — uneven muscle recruitment")
        for error in errors:
            if error.severity == "CRITICAL":
                explanation.append(f"CRITICAL: {error.error_type}")
            elif error.severity == "HIGH":
                explanation.append(f"HIGH risk: {error.error_type}")

    return grade, risk_score, explanation

if __name__ == "__main__":
    print("Testing grading system...\n")

    features_f = {'avg_elbow_angle': 125, 'extreme_flare': 1, 'bar_vertical_tilt': 0.02,
                  'elbow_angle_diff': 5, 'alignment_score': 0.9, 'symmetry_score': 0.03,
                  'shoulder_height_diff': 0.02}
    errors_f = [type('E', (), {'severity': 'CRITICAL', 'error_type': 'Extreme Elbow Flare'})()]
    grade, risk, exp = get_improved_grade_with_explanation(features_f, 1, 0.95, errors_f)
    print(f"Test 1 (Extreme flare, model says CORRECT): Grade={grade}, Risk={risk:.0f}")
    print(f"  Explanation: {exp}\n")

    features_bar = {'avg_elbow_angle': 80, 'extreme_flare': 0, 'bar_vertical_tilt': 0.12,
                    'elbow_angle_diff': 5, 'alignment_score': 0.9, 'symmetry_score': 0.05,
                    'shoulder_height_diff': 0.03}
    errors_bar = [type('E', (), {'severity': 'HIGH', 'error_type': 'Uneven Bar Path'})()]
    grade2, risk2, exp2 = get_improved_grade_with_explanation(features_bar, 1, 0.8, errors_bar)
    print(f"Test 2 (Uneven bar, model says CORRECT): Grade={grade2}, Risk={risk2:.0f}")
    print(f"  Explanation: {exp2}\n")

    features_ok = {'avg_elbow_angle': 65, 'extreme_flare': 0, 'bar_vertical_tilt': 0.02,
                   'elbow_angle_diff': 3, 'alignment_score': 0.95, 'symmetry_score': 0.02,
                   'shoulder_height_diff': 0.01}
    grade3, risk3, exp3 = get_improved_grade_with_explanation(features_ok, 1, 0.92, [])
    print(f"Test 3 (Good form): Grade={grade3}, Risk={risk3:.0f}")
    print(f"  Explanation: {exp3}")
