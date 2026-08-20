import cv2
import numpy as np
from collections import deque

class SeatbeltDetector:
    def __init__(self, history_size=5):
        # Multi-frame smoothing (prevents flickering detection)
        self.history = deque(maxlen=history_size)

    def detect(self, frame, face_landmarks, W, H):
        """
        HIGH ACCURACY SEATBELT DETECTION.
        Returns True if seatbelt WORN.
        """
        try:
            # ---------------------------------------------------------
            # 1) ROI ESTIMATION
            # ---------------------------------------------------------
            if face_landmarks:
                chin_y = max(lm.y for lm in face_landmarks.landmark)
                start_y = int(chin_y * H)
                end_y = H
            else:
                start_y = int(H * 0.35)
                end_y = H

            roi = frame[start_y:end_y]
            if roi.size == 0:
                self.history.append(False)
                return False

            # ---------------------------------------------------------
            # 2) PREPROCESSING
            # ---------------------------------------------------------
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # CLAHE for better belt visibility
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Reduce fabric noise
            blur = cv2.GaussianBlur(enhanced, (7, 7), 0)

            # Edges with stronger separation
            edges = cv2.Canny(blur, 40, 120, L2gradient=True)

            # ---------------------------------------------------------
            # 3) HOUGH LINE DETECTION with improved parameters
            # ---------------------------------------------------------
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=30, # Lowered threshold
                minLineLength=60, maxLineGap=40 # Shorter lines, larger gap
            )

            detected = False
            diagonal_lines = []
            
            if lines is not None:
                for (x1, y1, x2, y2) in lines[:, 0]:

                    # GLOBAL draw (red)
                    cv2.line(frame, (x1, y1 + start_y), (x2, y2 + start_y),
                             (0, 0, 255), 1)

                    # Skip nearly vertical/horizontal lines
                    if x2 == x1:
                        continue

                    slope = (y2 - y1) / (x2 - x1)
                    angle = abs(np.degrees(np.arctan(slope)))

                    # -----------------------------------------
                    # ACCEPT only realistic seatbelt angles
                    # (Stricter angle window = more accuracy)
                    # -----------------------------------------
                    # Widened range: 25 to 85
                    if 25 < angle < 85:
                        length = np.linalg.norm([x2 - x1, y2 - y1])

                        # Only meaningful lines
                        if length > 80: # Relaxed length check
                            diagonal_lines.append((x1, y1, x2, y2, angle, length))

            # ---------------------------------------------------------
            # 5) CLUSTERING LINES (Remove cloth folds)
            # Seatbelt lines tend to be parallel!
            # ---------------------------------------------------------
            cluster = []
            if len(diagonal_lines) > 0:
                angles = np.array([d[4] for d in diagonal_lines])
                angle_mean = np.mean(angles)
                cluster = [d for d in diagonal_lines if abs(d[4] - angle_mean) < 15] # Increased tolerance

            # ---------------------------------------------------------
            # 6) COLOR / TEXTURE VERIFICATION (IMPORTANT)
            # Seatbelts are usually dark, low-pattern
            # ---------------------------------------------------------
            belt_candidate_score = 0
            
            # If we have good lines, check them
            if len(cluster) >= 1: # Allow even 1 strong line
                for (x1, y1, x2, y2, angle, length) in cluster:
                    # Extract belt patch
                    patch = roi[min(y1, y2):max(y1, y2),
                                min(x1, x2):max(x1, x2)]

                    if patch.size == 0:
                        continue

                    # Mean color evaluation (belt is usually dark)
                    mean_val = np.mean(patch)

                    # Low variance means belt-like flat color
                    texture_var = np.var(patch)

                    # Seatbelt rules:
                    # - Darker region (Relaxed: < 200 for light belts)
                    # - Lower texture variation (Relaxed: < 1500 for patterned belts)
                    if mean_val < 200 and texture_var < 1500:
                        belt_candidate_score += 1
                        break # One good line is enough to get a point

            # ---------------------------------------------------------
            # 7) SHOULDER COLOR CHECK (User Requested)
            # Check for black/dark color near shoulder
            # ---------------------------------------------------------
            shoulder_roi_y = int(H * 0.6)
            shoulder_roi_h = int(H * 0.9)
            
            h_roi, w_roi = roi.shape[:2]
            
            # Left Shoulder Area (in ROI)
            l_shoulder = roi[int(h_roi*0.5):, :int(w_roi*0.3)]
            # Right Shoulder Area (in ROI)
            r_shoulder = roi[int(h_roi*0.5):, int(w_roi*0.7):]
            
            shoulder_detected = False
            
            # DEBUG: Draw Shoulder ROIs on frame
            ls_y1 = start_y + int(h_roi*0.5)
            ls_x1 = 0
            ls_x2 = int(w_roi*0.3)
            ls_y2 = start_y + h_roi
            cv2.rectangle(frame, (ls_x1, ls_y1), (ls_x2, ls_y2), (255, 0, 0), 1)
            
            rs_y1 = start_y + int(h_roi*0.5)
            rs_x1 = int(w_roi*0.7)
            rs_x2 = w_roi
            rs_y2 = start_y + h_roi
            cv2.rectangle(frame, (rs_x1, rs_y1), (rs_x2, rs_y2), (255, 0, 0), 1)

            # Chest Area (Center of ROI - likely shirt)
            chest_roi = roi[int(h_roi*0.2):int(h_roi*0.8), int(w_roi*0.3):int(w_roi*0.7)]
            chest_brightness = 150 # Default fallback
            if chest_roi.size > 0:
                chest_gray = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)
                chest_brightness = np.mean(chest_gray)
            
            # Draw Chest ROI for debug
            cv2.rectangle(frame, (int(w_roi*0.3), start_y + int(h_roi*0.2)), (int(w_roi*0.7), start_y + int(h_roi*0.8)), (0, 255, 255), 1)
            cv2.putText(frame, f"Chest: {int(chest_brightness)}", (int(w_roi*0.3), start_y + int(h_roi*0.2) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            for i, sh_patch in enumerate([l_shoulder, r_shoulder]):
                if sh_patch.size == 0: continue
                
                sh_gray = cv2.cvtColor(sh_patch, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(sh_gray)
                
                # Print brightness for debugging
                print(f"Shoulder {i}: {avg_brightness:.1f} | Chest: {chest_brightness:.1f}")
                
                if i == 0: # Left
                    cv2.putText(frame, f"{int(avg_brightness)}", (ls_x1, ls_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else: # Right
                    cv2.putText(frame, f"{int(avg_brightness)}", (rs_x1, rs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # LOGIC:
                # 1. Absolute Darkness (Black/Grey Belt) - Relaxed to < 160
                # 2. Relative Contrast (Shoulder is darker than Chest)
                
                is_dark_shoulder = avg_brightness < 160
                
                # Contrast check: Shoulder is 10% darker than chest (Relaxed from 25%)
                is_contrast_shoulder = avg_brightness < (chest_brightness * 0.90)
                
                # Special Case: Dark Shirt (Chest < 80)
                if chest_brightness < 80:
                    is_contrast_shoulder = False 
                
                if is_dark_shoulder or is_contrast_shoulder: 
                    shoulder_detected = True
                    # Draw filled rectangle to show detection
                    if i == 0: # Left
                         cv2.rectangle(frame, (ls_x1, ls_y1), (ls_x2, ls_y2), (0, 255, 0), 2)
                    else: # Right
                         cv2.rectangle(frame, (rs_x1, rs_y1), (rs_x2, rs_y2), (0, 255, 0), 2)
            
            if shoulder_detected:
                belt_candidate_score += 1 

            # ---------------------------------------------------------
            # 8) CHEST DIAGONAL CHECK (New)
            # Look for a dark band across the chest ROI
            # ---------------------------------------------------------
            if chest_roi.size > 0:
                h_c, w_c = chest_roi.shape[:2]
                
                mask_diag1 = np.zeros((h_c, w_c), dtype=np.uint8)
                cv2.line(mask_diag1, (0, 0), (w_c, h_c), 255, thickness=int(w_c*0.3)) # Thick diagonal
                
                mask_diag2 = np.zeros((h_c, w_c), dtype=np.uint8)
                cv2.line(mask_diag2, (w_c, 0), (0, h_c), 255, thickness=int(w_c*0.3)) # Other diagonal
                
                chest_gray_check = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)
                
                mean_diag1 = cv2.mean(chest_gray_check, mask=mask_diag1)[0]
                mean_diag2 = cv2.mean(chest_gray_check, mask=mask_diag2)[0]
                
                # If diagonal is 5% darker than overall chest (Relaxed)
                if (mean_diag1 < chest_brightness * 0.95) or (mean_diag2 < chest_brightness * 0.95):
                    belt_candidate_score += 1
                    # Draw diagonal on screen for debug
                    if mean_diag1 < mean_diag2:
                         cv2.line(frame, (int(w_roi*0.3), start_y + int(h_roi*0.2)), (int(w_roi*0.7), start_y + int(h_roi*0.8)), (0, 255, 0), 2)
                    else:
                         cv2.line(frame, (int(w_roi*0.7), start_y + int(h_roi*0.2)), (int(w_roi*0.3), start_y + int(h_roi*0.8)), (0, 255, 0), 2)

            # ---------------------------------------------------------
            # 8) DECISION
            # ---------------------------------------------------------
            if belt_candidate_score >= 1: # Relaxed: Just 1 strong indicator is enough (e.g. just shoulder or just lines)
                detected = True

                # Draw strongest lines in GREEN
                for (x1, y1, x2, y2, _, _) in cluster:
                    cv2.line(frame, (x1, y1 + start_y), (x2, y2 + start_y),
                             (0, 255, 0), 3)

            # ---------------------------------------------------------
            # 8) MULTI-FRAME SMOOTHING
            # ---------------------------------------------------------
            self.history.append(detected)
            confirmed = sum(self.history) > (len(self.history) * 0.6)

            return confirmed

        except Exception as e:
            print("Seatbelt detection error:", e)
            self.history.append(False)
            return False
