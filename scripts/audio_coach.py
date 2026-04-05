import time
from threading import Thread, Lock
from collections import deque
import subprocess
import platform

class AudioCoach:
    """
    Real-time audio coaching system with intelligent feedback timing
    Uses system-native TTS to avoid crashes
    """
    
    def __init__(self, rate=150, volume=0.9):
        """
        Initialize audio coach
        
        Args:
            rate: Speech rate (words per minute, default 150)
            volume: Volume level (0.0 to 1.0, default 0.9)
        """
        self.rate = rate
        self.volume = volume
        self.system = platform.system()
        
        # Thread-safe speaking flag
        self.is_speaking = False
        self.speech_lock = Lock()
        
        # Feedback tracking
        self.last_feedback_time = {}
        self.feedback_cooldown = 3.0
        self.feedback_queue = deque(maxlen=5)
        
        # Priority levels
        self.PRIORITY = {
            'CRITICAL': 1,
            'HIGH': 2,
            'MEDIUM': 3,
            'LOW': 4,
            'POSITIVE': 5
        }
        
        # Check if TTS is available
        self.tts_available = self._check_tts_available()
        
        if self.tts_available:
            print(" Audio system initialized successfully")
        else:
            print("  Audio system not available on this platform")
    
    def _check_tts_available(self):
        """Check if text-to-speech is available on this system"""
        try:
            if self.system == "Darwin":  # macOS
                # Test if 'say' command works
                subprocess.run(['say', '-v', '?'], capture_output=True, timeout=1)
                return True
            elif self.system == "Windows":
                # Windows has built-in TTS
                return True
            elif self.system == "Linux":
                # Check if espeak is available
                subprocess.run(['which', 'espeak'], capture_output=True, timeout=1)
                return True
        except:
            pass
        return False
    
    def speak_async(self, text):
        """
        Speak text asynchronously (non-blocking) using system TTS
        """
        if not self.tts_available:
            return
        
        def _speak():
            with self.speech_lock:
                try:
                    self.is_speaking = True
                    
                    if self.system == "Darwin":  # macOS
                        # Use macOS 'say' command (more reliable than pyttsx3)
                        rate_wpm = int(self.rate * 0.6)  # Convert to words per minute
                        subprocess.run(
                            ['say', '-r', str(rate_wpm), text],
                            timeout=5,
                            capture_output=True
                        )
                    
                    elif self.system == "Windows":
                        # Use Windows PowerShell TTS
                        ps_command = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = 0; $synth.Volume = 100; $synth.Speak("{text}")'
                        subprocess.run(
                            ['powershell', '-Command', ps_command],
                            timeout=5,
                            capture_output=True
                        )
                    
                    elif self.system == "Linux":
                        # Use espeak
                        subprocess.run(
                            ['espeak', text],
                            timeout=5,
                            capture_output=True
                        )
                    
                except subprocess.TimeoutExpired:
                    print("  Speech timeout - skipping")
                except Exception as e:
                    print(f"  Speech error: {e}")
                finally:
                    self.is_speaking = False
        
        # Start speaking in background thread
        thread = Thread(target=_speak, daemon=True)
        thread.start()
    
    def should_give_feedback(self, feedback_key):
        """
        Check if enough time has passed since last feedback
        """
        current_time = time.time()
        
        if feedback_key not in self.last_feedback_time:
            self.last_feedback_time[feedback_key] = current_time
            return True
        
        time_since_last = current_time - self.last_feedback_time[feedback_key]
        
        if time_since_last >= self.feedback_cooldown:
            self.last_feedback_time[feedback_key] = current_time
            return True
        
        return False
    
    def analyze_and_coach(self, features, prediction, errors, rep_count):
        """
        Analyze current state and provide appropriate coaching
        """
        if self.is_speaking or not self.tts_available:
            return
        
        # Priority-ordered feedback
        feedback_messages = []
        
        # 1. CRITICAL: Extreme elbow flare
        avg_elbow = features.get('avg_elbow_angle', 0)
        if avg_elbow > 120:
            if self.should_give_feedback('critical_elbow'):
                feedback_messages.append(('CRITICAL', "Stop! Elbows extremely flared. Risk of injury."))
        
        # 2. CRITICAL: Uneven bar (high tilt)
        bar_tilt = features.get('bar_vertical_tilt', 0) * 100
        if bar_tilt > 10:
            if self.should_give_feedback('critical_bar_tilt'):
                feedback_messages.append(('CRITICAL', "Warning! Bar is very tilted. Level it out now."))
        
        # 3. HIGH: Excessive elbow flare
        elif avg_elbow > 75:
            if self.should_give_feedback('high_elbow'):
                feedback_messages.append(('HIGH', "Elbows too wide. Tuck them closer to your body."))
        
        # 4. HIGH: Shoulder asymmetry
        shoulder_diff = features.get('shoulder_height_diff', 0) * 100
        if shoulder_diff > 8:
            if self.should_give_feedback('shoulder_asym'):
                if features.get('left_shoulder_y', 0) < features.get('right_shoulder_y', 0):
                    feedback_messages.append(('HIGH', "Left shoulder is higher. Balance your grip."))
                else:
                    feedback_messages.append(('HIGH', "Right shoulder is higher. Balance your grip."))
        
        # 5. HIGH: Bar tilted
        elif bar_tilt > 5:
            if self.should_give_feedback('bar_tilt'):
                feedback_messages.append(('HIGH', "Bar is tilted. Keep it level."))
        
        # 6. MEDIUM: Elbow asymmetry
        elbow_angle_diff = features.get('elbow_angle_diff', 0)
        if elbow_angle_diff > 20:
            if self.should_give_feedback('elbow_asym'):
                feedback_messages.append(('MEDIUM', "Elbows are uneven. Move both arms equally."))
        
        # 7. MEDIUM: Bar off center
        bar_center = features.get('bar_center_offset', 0) * 100
        if bar_center > 15:
            if self.should_give_feedback('bar_center'):
                direction = "left" if features.get('wrist_x', 0.5) < 0.5 else "right"
                feedback_messages.append(('MEDIUM', f"Bar shifted {direction}. Center yourself."))
        
        # 8. MEDIUM: Poor scapular retraction
        retraction = features.get('retraction_normalized', 0)
        if retraction > 0.15:
            if self.should_give_feedback('retraction'):
                feedback_messages.append(('MEDIUM', "Pull shoulder blades together and down."))
        
        # 9. LOW: General symmetry issues
        symmetry_score = features.get('symmetry_score', 0)
        if symmetry_score > 0.12 and not feedback_messages:
            if self.should_give_feedback('symmetry'):
                feedback_messages.append(('LOW', "Focus on symmetry. Both sides should move equally."))
        
        # 10. POSITIVE: Good form feedback (every 3 reps)
        if prediction == 1 and symmetry_score < 0.08 and avg_elbow < 75:
            if rep_count > 0 and rep_count % 3 == 0:
                if self.should_give_feedback('positive'):
                    positive_messages = [
                        "Excellent form. Keep it up.",
                        "Perfect technique. Stay controlled.",
                        "Great symmetry. Maintain this form.",
                        "Good control. Nice work."
                    ]
                    msg_index = (rep_count // 3) % len(positive_messages)
                    feedback_messages.append(('POSITIVE', positive_messages[msg_index]))
        
        # Speak highest priority message
        if feedback_messages:
            feedback_messages.sort(key=lambda x: self.PRIORITY[x[0]])
            priority, message = feedback_messages[0]
            
            print(f" AUDIO COACH [{priority}]: {message}")
            self.speak_async(message)
    
    def rep_completed_feedback(self, rep_analysis):
        """
        Provide feedback when rep is completed
        """
        if self.is_speaking or not self.tts_available:
            return
        
        grade = rep_analysis.get_quality_grade()
        
        if grade == 'A':
            message = "Perfect rep!"
        elif grade == 'B':
            message = "Good rep. Keep it up."
        elif grade == 'C':
            message = "Rep completed. Watch your form."
        elif grade == 'D':
            message = "Rep completed. Focus on symmetry."
        else:
            message = "Poor technique detected. Reduce weight."
        
        print(f" REP FEEDBACK: {message}")
        self.speak_async(message)
    
    def set_completed_feedback(self, summary):
        """
        Provide summary when set is completed
        """
        if not self.tts_available:
            return
        
        total = summary['total_reps']
        correct = summary['correct_reps']
        accuracy = summary['accuracy_rate']
        
        if accuracy == 100:
            message = f"Set complete! {total} perfect reps. Excellent work!"
        elif accuracy >= 80:
            message = f"Set complete! {correct} out of {total} reps correct. Good job!"
        else:
            message = f"Set complete! {correct} out of {total} reps correct. Focus on form."
        
        print(f" SET SUMMARY: {message}")
        self.speak_async(message)
    
    def change_voice(self, voice_index=0):
        """
        Change TTS voice (not supported with system TTS)
        """
        if self.system == "Darwin":
            voices = ['Alex', 'Samantha', 'Victoria', 'Karen']
            # Could implement voice selection for macOS 'say' command
            print(f"ℹ  Voice selection: Using system default")
        else:
            print(f"ℹ  Voice selection not available on {self.system}")
    
    def set_speech_rate(self, rate):
        """
        Adjust speech speed (100-200 recommended)
        """
        self.rate = rate
        print(f" Speech rate set to {rate} WPM")
    
    def set_volume(self, volume):
        """
        Adjust volume (0.0 to 1.0)
        """
        self.volume = volume
        print(f" Volume set to {int(volume * 100)}%")
    
    def set_cooldown(self, seconds):
        """
        Adjust feedback cooldown period
        """
        self.feedback_cooldown = seconds
        print(f" Feedback cooldown set to {seconds} seconds")