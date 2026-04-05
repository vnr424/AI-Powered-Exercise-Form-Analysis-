import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PersonIsolator:
    """
    Detect person in frame and return isolated region for pose detection
    """
    
    def __init__(self, method='yolo'):
        """
        Initialize person detector
        
        Args:
            method: 'yolo' or 'mediapipe'
        """
        self.method = method
        
        if method == 'yolo':
            # YOLOv8 nano model (fast, accurate)
            try:
                self.detector = YOLO('yolov8n.pt')
                print("YOLOv8 person detector loaded")
            except:
                print("Warning: YOLOv8 not available, falling back to mediapipe")
                self.method = 'mediapipe'
        
        if self.method == 'mediapipe':
            # MediaPipe Object Detection
            base_options = python.BaseOptions(
                model_asset_path='models/efficientdet_lite0.tflite'
            )
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                max_results=5,
                score_threshold=0.3
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
            print("MediaPipe object detector loaded")
    
    def get_person_roi(self, frame, timestamp_ms=0):
        """
        Detect person and return region of interest
        
        Returns:
            roi_frame: Cropped frame containing only person
            bbox: (x, y, w, h) bounding box coordinates
            scale_factor: For converting coordinates back to original frame
        """
        h, w = frame.shape[:2]
        
        if self.method == 'yolo':
            return self._yolo_detect(frame)
        else:
            return self._mediapipe_detect(frame, timestamp_ms)
    
    def _yolo_detect(self, frame):
        """YOLOv8 person detection"""
        results = self.detector(frame, classes=[0], verbose=False)  # class 0 = person
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None, None
        
        # Get largest person (by area)
        boxes = results[0].boxes
        areas = boxes.xywh[:, 2] * boxes.xywh[:, 3]
        max_idx = areas.argmax()
        
        box = boxes.xywh[max_idx].cpu().numpy()
        cx, cy, bw, bh = box
        
        # Add 20% padding
        padding = 0.2
        bw_padded = bw * (1 + padding)
        bh_padded = bh * (1 + padding)
        
        x1 = int(max(0, cx - bw_padded/2))
        y1 = int(max(0, cy - bh_padded/2))
        x2 = int(min(frame.shape[1], cx + bw_padded/2))
        y2 = int(min(frame.shape[0], cy + bh_padded/2))
        
        roi = frame[y1:y2, x1:x2]
        bbox = (x1, y1, x2-x1, y2-y1)
        
        return roi, bbox, (x1, y1)
    
    def _mediapipe_detect(self, frame, timestamp_ms):
        """MediaPipe object detection"""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if not detection_result.detections:
            return None, None, None
        
        # Find person (category 0)
        person_detections = [d for d in detection_result.detections 
                           if d.categories[0].index == 0]
        
        if not person_detections:
            return None, None, None
        
        # Get largest
        detection = max(person_detections, 
                       key=lambda d: d.bounding_box.width * d.bounding_box.height)
        
        bbox = detection.bounding_box
        x1 = bbox.origin_x
        y1 = bbox.origin_y
        w = bbox.width
        h = bbox.height
        
        # Add padding
        padding = 0.2
        x1 = int(max(0, x1 - w*padding/2))
        y1 = int(max(0, y1 - h*padding/2))
        w = int(w * (1 + padding))
        h = int(h * (1 + padding))
        
        x2 = min(frame.shape[1], x1 + w)
        y2 = min(frame.shape[0], y1 + h)
        
        roi = frame[y1:y2, x1:x2]
        bbox = (x1, y1, w, h)
        
        return roi, bbox, (x1, y1)

def integrate_person_filter(frame, pose_detector, person_isolator, timestamp_ms):
    """
    Integrated pipeline: Person detection -> Pose detection
    SIMPLIFIED: Detect person region but run pose on FULL frame for stability
    
    Returns:
        detection_result: Pose landmarks 
        person_bbox: Bounding box of detected person (for display only)
    """
    # Detect person bounding box (for verification only)
    roi, bbox, offset = person_isolator.get_person_roi(frame, timestamp_ms)
    
    # ALWAYS run pose detection on full frame (more stable)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
    
    # If person detected, verify landmarks are within person bbox
    if bbox and detection_result.pose_landmarks:
        x, y, w, h = bbox
        h_frame, w_frame = frame.shape[:2]
        
        # Check if main landmarks are within person region
        landmarks = detection_result.pose_landmarks[0]
        nose = landmarks[0]
        
        nose_x = int(nose.x * w_frame)
        nose_y = int(nose.y * h_frame)
        
        # If nose is far outside person bbox, likely wrong detection
        if not (x - 50 < nose_x < x + w + 50 and y - 50 < nose_y < y + h + 50):
            # Landmarks detected on wrong object - return empty
            detection_result.pose_landmarks[:] = []
    
    return detection_result, bbox

# Usage example
if __name__ == "__main__":
    # Initialize
    person_isolator = PersonIsolator(method='yolo')  # or 'mediapipe'
    
    # In your main loop, replace:
    # detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    # With:
    # detection_result, person_bbox = integrate_person_filter(
    #     frame, detector, person_isolator, timestamp_ms
    # )
    
    # Optionally draw person bbox for debugging
    # if person_bbox:
    #     x, y, w, h = person_bbox
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)