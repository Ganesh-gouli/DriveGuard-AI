import cv2
import numpy as np


class SunglassesDetector:
    """
    Robust sunglasses detection using MediaPipe Face Mesh landmarks.
    Uses brightness ratio + texture variance difference between eyes & skin.
    """

    def __init__(self, ratio_thresh=0.32, extreme_dark_thresh=0.15, var_thresh=10, debug=False):
        self.ratio_thresh = ratio_thresh
        self.extreme_dark_thresh = extreme_dark_thresh
        self.var_thresh = var_thresh
        self.debug = debug

        # Pre-store indices (performance improvement)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.SKIN = [168, 6, 197, 195]
        self.BRIDGE = [6, 168, 197, 195] # Nose bridge area (between eyes)

    def get_roi_stats(self, frame, landmarks, indices, w, h):
        """
        Extract region brightness + variance using polygon mask.
        Returns (mean, stddev).
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = []

        for idx in indices:
            lm = landmarks.landmark[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))

        if len(points) < 3:
            return 0, 0  # invalid polygon

        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val, std_val = cv2.meanStdDev(gray, mask=mask)

        return float(mean_val[0][0]), float(std_val[0][0])

    def detect(self, frame, face_landmarks, w, h):
        """
        Returns True if sunglasses detected, False otherwise.
        """

        try:
            # ROI stats
            left_m, left_std = self.get_roi_stats(frame, face_landmarks, self.LEFT_EYE, w, h)
            right_m, right_std = self.get_roi_stats(frame, face_landmarks, self.RIGHT_EYE, w, h)
            skin_m, skin_std = self.get_roi_stats(frame, face_landmarks, self.SKIN, w, h)
            bridge_m, bridge_std = self.get_roi_stats(frame, face_landmarks, self.BRIDGE, w, h)

            # Average values
            eye_bright = (left_m + right_m) / 2
            eye_var = (left_std + right_std) / 2

            # Prevent division issues
            if skin_m < 10:  
                if self.debug:
                    print("Lighting too dark → Not detecting sunglasses.")
                return False  

            brightness_ratio = eye_bright / skin_m
            bridge_ratio = bridge_m / skin_m

            if self.debug:
                print(f"[SUNGLASSES DEBUG]")
                print(f"Eye Brightness     : {eye_bright:.2f}")
                print(f"Skin Brightness    : {skin_m:.2f}")
                print(f"Bridge Brightness  : {bridge_m:.2f}")
                print(f"Brightness Ratio   : {brightness_ratio:.2f}")
                print(f"Bridge Ratio       : {bridge_ratio:.2f}")
                print(f"Eye Variance       : {eye_var:.2f}")
                print("-" * 40)

            # --- DECISION LOGIC ---

            # Case 1: Very dark region → high confidence sunglasses
            if brightness_ratio < self.extreme_dark_thresh:
                return True

            # Case 2: moderately dark + low texture = sunglasses
            if brightness_ratio < self.ratio_thresh and eye_var < self.var_thresh:
                return True
            
            # Case 3: Bridge Check (Dark frames on nose)
            # If bridge is significantly darker than skin (e.g. < 0.6 ratio)
            if bridge_ratio < 0.6:
                 if self.debug: print("Bridge check passed (Dark Frames)")
                 return True

            return False

        except Exception as e:
            if self.debug:
                print(f"SunglassesDetector Error: {e}")
            return False
