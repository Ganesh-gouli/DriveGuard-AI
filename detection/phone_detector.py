import time
import cv2
import numpy as np

class PhoneDetector:
    def __init__(self, model_path="yolov8n.pt"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.available = True
        except Exception as e:
            print(f"PhoneDetector init error: {e}")
            self.available = False

        # Frame history for temporal smoothing
        self.history = []
        self.history_size = 45  # ~1.5–2 sec history

        # Thresholds
        self.CONF_THRESHOLD = 0.40
        self.IOU_FACE_PHONE_THRESHOLD = 0.10
        self.DIST_FACE_THRESHOLD = 0.22
        self.DIST_HAND_THRESHOLD = 0.15

    # ---------------------------------------------------------
    # Utility: IOU
    # ---------------------------------------------------------
    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        if boxAArea + boxBArea - interArea == 0:
            return 0

        return interArea / (boxAArea + boxBArea - interArea)

    # ---------------------------------------------------------
    # Detection
    # ---------------------------------------------------------
    def detect(self, frame):
        if not self.available:
            return [], False, []

        h, w = frame.shape[:2]

        results = self.model(frame, conf=self.CONF_THRESHOLD, verbose=False)

        phones = []
        faces = []
        hands = []

        # Parse detections
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                xyxy = box.xyxy[0].tolist()
                cx = (xyxy[0] + xyxy[2]) / 2 / w
                cy = (xyxy[1] + xyxy[3]) / 2 / h

                det = {"box": xyxy, "center": (cx, cy), "conf": conf}

                if cls_name in ["cell phone", "cell_phone", "phone"]:
                    phones.append(det)

                elif cls_name in ["person", "driver_face", "face"]:
                    faces.append(det)

                elif cls_name in ["hand", "palm"]:
                    hands.append(det)

        # ---------------------------------------------------------
        # Enhanced Phone Usage Logic
        # ---------------------------------------------------------
        phone_in_use_frame = False

        for phone in phones:

            # (1) Check IOU overlap with face (very strong indicator)
            for face in faces:
                iou_value = self.iou(phone["box"], face["box"])
                if iou_value > self.IOU_FACE_PHONE_THRESHOLD:
                    phone_in_use_frame = True
                    break

            if phone_in_use_frame:
                break

            # (2) Distance-based rule
            for face in faces:
                dist = np.linalg.norm(np.array(phone["center"]) - np.array(face["center"]))
                if dist < self.DIST_FACE_THRESHOLD:
                    phone_in_use_frame = True
                    break

            if phone_in_use_frame:
                break

            # (3) Check if in hand
            for hand in hands:
                # Distance check
                dist = np.linalg.norm(np.array(phone["center"]) - np.array(hand["center"]))
                if dist < self.DIST_HAND_THRESHOLD:
                    phone_in_use_frame = True
                    # break # Don't break yet, check IOU too

                # IOU Check (Hand holding phone)
                iou_hand = self.iou(phone["box"], hand["box"])
                if iou_hand > 0.05: # Any significant overlap
                     phone_in_use_frame = True
                     break
            
            if phone_in_use_frame:
                break

        # ---------------------------------------------------------
        # (4) Second-Stage Verification (noisy false positives filter)
        # Phone detection is confirmed only if:
        # - phone box has a rectangular shape (portrait OR landscape)
        # - brightness/color distribution matches a typical phone
        # ---------------------------------------------------------
        if phone_in_use_frame and len(phones) > 0:
            for ph in phones:
                x1, y1, x2, y2 = map(int, ph["box"])
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    h_ph, w_ph = crop.shape[:2]
                    
                    # Check rectangularity (Portrait OR Landscape)
                    # A phone is rarely perfectly square.
                    long_side = max(h_ph, w_ph)
                    short_side = min(h_ph, w_ph)
                    aspect_ratio = long_side / max(short_side, 1)

                    # Relaxed check: Must be somewhat rectangular (> 1.2 ratio)
                    # But if it's too square (< 1.2), it might be a false positive (like a cup or post-it)
                    # HOWEVER, user wants "improved" detection, so let's be lenient.
                    # Let's just reject extremely flat/thin things or perfect squares if needed.
                    # Actually, let's REMOVE the strict aspect ratio check for now to allow more detections.
                    # if aspect_ratio < 1.2: 
                    #    phone_in_use_frame = False

                    # Detect reflective surface (high contrast)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    contrast = np.std(gray)

                    # Relaxed Contrast Threshold (was 20)
                    if contrast < 10:  # too uniform (not a phone)
                        phone_in_use_frame = False

        # ---------------------------------------------------------
        # Temporal Smoothing (Multi-frame validation)
        # ---------------------------------------------------------
        self.history.append(phone_in_use_frame)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        confirmed = sum(self.history) > (0.45 * len(self.history))

        return results, confirmed, phones
