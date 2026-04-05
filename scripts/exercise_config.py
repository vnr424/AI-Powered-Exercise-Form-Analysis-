class ExerciseConfig:
    """
    Configuration for different exercises
    """
    
    EXERCISES = {
        'bench_press': {
            'name': 'Bench Press',
            'description': 'Lying flat bench press',
            'required_landmarks': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
            'optional_landmarks': ['left_hip', 'right_hip'],  # Not visible in bench press
            'min_visibility_threshold': 0.5,
            'use_hip_features': False,  # Don't use hip-based features
            'specific_checks': {
                'check_elbow_flare': True,
                'check_bar_path': True,
                'check_symmetry': True,
                'check_scapular_retraction': False,  # Can't measure accurately without hips
                'check_hip_tilt': False,  # Hips not visible
            },
            'elbow_angle_range': (45, 75),  # Optimal elbow angle for bench press
            'critical_elbow_angle': 120,
            'camera_position': 'Position camera in front of bench at chest level',
        },
        
        'shoulder_press': {
            'name': 'Standing Barbell Shoulder Press',
            'description': 'Standing overhead press',
            'required_landmarks': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip'],
            'optional_landmarks': [],
            'min_visibility_threshold': 0.5,
            'use_hip_features': True,  # Use hip-based features
            'specific_checks': {
                'check_elbow_flare': True,
                'check_bar_path': True,
                'check_symmetry': True,
                'check_scapular_retraction': True,
                'check_hip_tilt': True,
            },
            'elbow_angle_range': (60, 90),  # Different optimal range for shoulder press
            'critical_elbow_angle': 120,
            'camera_position': 'Position camera to show full body from front',
        }
    }
    
    @classmethod
    def get_exercise_config(cls, exercise_type):
        """Get configuration for specific exercise"""
        return cls.EXERCISES.get(exercise_type, cls.EXERCISES['bench_press'])
    
    @classmethod
    def get_available_exercises(cls):
        """Get list of available exercises"""
        return [(key, config['name']) for key, config in cls.EXERCISES.items()]
    
    @classmethod
    def check_landmark_visibility(cls, pose_landmarks, exercise_type):
        """
        Check if required landmarks are visible for the exercise
        Returns: (is_valid, missing_landmarks)
        """
        config = cls.get_exercise_config(exercise_type)
        required = config['required_landmarks']
        threshold = config['min_visibility_threshold']
        
        LANDMARKS = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24
        }
        
        lm = pose_landmarks[0]
        missing = []
        
        for landmark_name in required:
            idx = LANDMARKS[landmark_name]
            landmark = lm[idx]
            
            if landmark.visibility < threshold:
                missing.append(landmark_name)
        
        is_valid = len(missing) == 0
        return is_valid, missing