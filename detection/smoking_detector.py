import numpy as np
import cv2
from collections import deque

class SmokingDetector:
    def __init__(self, model_path='driver_custom_model/yolov8n_custom_v1/weights/best.pt', history_size=6):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
        except:
            print("Smoking detection model unavailable. Using default.")
            try:
                self.model = YOLO('yolov8n.pt')
                self.available = True
                self.target_class_id = 79 # Fallback to toothbrush
            except:
                self.available = False

        # Custom Model: 0 = cigarette
        # Default Model: 79 = toothbrush
        if 'custom' in model_path:
             self.target_class_id = 0
        else:
             self.target_class_id = 79
        
        self.conf_threshold = 0.25

        # Temporal smoothing history
        self.history = deque(maxlen=history_size)

    def detect(self, frame, face_landmarks, W, H, debug=False, hand_landmarks_list=None):
        if not self.available:
            return False, []

        try:
            if not face_landmarks:
                self.history.append(False)
                return False, []

            # ------------------------------
            # 1) Get mouth region
            # ------------------------------
            mouth_indices = [61, 291, 0, 17]
            mouth_pts = [(face_landmarks.landmark[i].x * W,
                          face_landmarks.landmark[i].y * H)
                         for i in mouth_indices]

            mouth_pts = np.array(mouth_pts)
            mouth_center = np.mean(mouth_pts, axis=0)

            mouth_width = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
            proximity_thresh = mouth_width * 3.2
            
            smoking_detected_frame = False
            detected_objects = []

            # ------------------------------
            # 2) HAND GESTURE CHECK (MediaPipe)
            # ------------------------------
            if hand_landmarks_list:
                for hand_landmarks in hand_landmarks_list:
                    # Index Finger Tip (ID 8)
                    index_tip = hand_landmarks.landmark[8]
                    ix, iy = index_tip.x * W, index_tip.y * H
                    
                    # Middle Finger Tip (ID 12)
                    middle_tip = hand_landmarks.landmark[12]
                    mx, my = middle_tip.x * W, middle_tip.y * H
                    
                    # Check distance to mouth
                    dist_index = np.linalg.norm(np.array([ix, iy]) - mouth_center)
                    dist_middle = np.linalg.norm(np.array([mx, my]) - mouth_center)
                    
                    # Threshold: e.g., 50-80 pixels (depends on resolution, use mouth_width relative)
                    hand_thresh = mouth_width * 2.0 
                    
                    if dist_index < hand_thresh or dist_middle < hand_thresh:
                        smoking_detected_frame = True
                        if debug:
                            print(f"Hand near mouth! Index: {dist_index:.1f}, Middle: {dist_middle:.1f}")
                        
                        # Create a dummy box for visualization around the fingers
                        x1 = int(min(ix, mx) - 20)
                        y1 = int(min(iy, my) - 20)
                        x2 = int(max(ix, mx) + 20)
                        y2 = int(max(iy, my) + 20)
                        
                        # Create a dummy object with xyxy attribute
                        class DummyBox:
                            def __init__(self, xyxy):
                                self.xyxy = np.array([xyxy])
                        
                        detected_objects.append(DummyBox([x1, y1, x2, y2]))
                        break # Found a hand near mouth

            # ------------------------------
            # 3) YOLO detection (Existing Logic)
            # ------------------------------
            if not smoking_detected_frame: # Only run YOLO if hand check didn't trigger (or run both?)
                results = self.model(frame, verbose=False,
                                     conf=self.conf_threshold)
    
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls != self.target_class_id:
                            continue
    
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w_box = x2 - x1
                        h_box = y2 - y1
                        obj_center = np.array([(x1 + x2) / 2,
                                               (y1 + y2) / 2])
    
                        # ------------------------------
                        # SHAPE FILTER (RELAXED)
                        # ------------------------------
                        aspect_ratio = max(w_box, h_box) / max(1, min(w_box, h_box))
                        if aspect_ratio < 1.5:   
                            continue
    
                        # ------------------------------
                        # Reject oversized objects
                        # ------------------------------
                        if w_box > 200 or h_box > 200: 
                            continue
    
                        # ------------------------------
                        # Proximity Check
                        # ------------------------------
                        dist = np.linalg.norm(obj_center - mouth_center)
                        if dist > proximity_thresh:
                            continue
    
                        detected_objects.append(box)
                        smoking_detected_frame = True
                        if debug:
                            print(f"Smoking Object Accepted! AspectRatio: {aspect_ratio:.2f}")

            # ------------------------------
            # 4) TEMPORAL SMOOTHING
            # ------------------------------
            self.history.append(smoking_detected_frame)
            final_decision = sum(self.history) >= (len(self.history) * 0.66)

            return final_decision, detected_objects

        except Exception as e:
            print("Smoking detection error:", e)
            self.history.append(False)
            return False, []
