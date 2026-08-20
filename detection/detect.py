import cv2
import mediapipe as mp
import time
import os
import json
from eye_detection import avg_ear
from mouth_detection import mar_from_landmarks
from alarm import play_alarm
from notifier import send_alert
from head_pose import get_head_pose, is_head_nodding
from low_light_enhancer import LowLightEnhancer
from smoking_detector import SmokingDetector

# Live Data File
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_DATA_FILE = os.path.join(BASE_DIR, "backend", "static", "live_data.json")

def write_live_data(eye_score, yawn_score, tilt_score, overall_risk, driving_stats=None, seatbelt_status="WORN", phone_status="NOT_DETECTED", sunglasses_status="NOT_DETECTED", smoking_status="NOT_DETECTED"):
    data = {
        "eye_score": eye_score,
        "yawn_score": yawn_score,
        "tilt_score": tilt_score,
        "overall_risk": overall_risk,
        "seatbelt_status": seatbelt_status,
        "phone_status": phone_status,
        "sunglasses_status": sunglasses_status,
        "smoking_status": smoking_status,
        "timestamp": time.time()
    }
    if driving_stats:
        data.update(driving_stats)
        
    try:
        with open(LIVE_DATA_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("Error writing live data:", e)

EAR_THRESH = 0.23
EAR_CONSEC_FRAMES = 18   # ~0.6-1.0s at 20-30 FPS
MAR_THRESH = 0.6
MAR_CONSEC_FRAMES = 15

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# helper to convert normalized landmarks to pixel coords
def landmarks_to_points(landmarks, image_w, image_h):
    pts = []
    for lm in landmarks.landmark:
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        pts.append((x, y))
    return pts

import argparse

def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)  # use command line argument
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}. Try specifying a different index with --camera <index>")
        return
    print(f"Camera {args.camera} opened successfully")

    closed_frames = 0
    yawning_frames = 0
    nodding_frames = 0
    turning_frames = 0
    last_alert_time = 0
    alarm_has_played = False
    seatbelt_status = "WORN" # Default status

    DRIVER_NAME = "Ganesh"  # change as needed
    
    # Driving Timer
    start_time = time.time()
    BREAK_INTERVAL_MIN = 90 # 1.5 hours

    # Initialize Detectors
    from seatbelt_detector import SeatbeltDetector
    seatbelt_detector = SeatbeltDetector()
    seatbelt_frame_count = 0
    last_seatbelt_alert_time = 0
    seatbelt_worn_frames = 0
    seatbelt_not_worn_frames = 0

    from sunglasses_detector import SunglassesDetector
    sunglasses_detector = SunglassesDetector()
    last_sunglasses_alert_time = 0
    is_sunglasses_detected = False
    sunglasses_frame_count = 0

    from phone_detector import PhoneDetector
    phone_detector = PhoneDetector()
    phone_status = "NOT_DETECTED"

    # Initialize Low Light Enhancer
    enhancer = LowLightEnhancer()
    
    # Initialize Smoking Detector
    smoking_detector = SmokingDetector()
    smoking_frame_count = 0
    smoking_status = "NOT_DETECTED"

    try:
        print("Starting detection loop...")
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from camera. Is it being used by another app?")
                break
                
            # Apply Low Light Enhancement (Night Vision)
            frame = enhancer.enhance(frame)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status_text = "Active"
            color = (0,255,0)
            
            # Default values for live data
            eye_score = 100
            yawn_score = 0
            tilt_score = 0
            overall_risk = 0
            
            # Calculate Driving Stats
            elapsed_sec = int(time.time() - start_time)
            hours = elapsed_sec // 3600
            mins = (elapsed_sec % 3600) // 60
            driving_time_str = f"{hours}h {mins}m"
            
            elapsed_min = elapsed_sec // 60
            remaining_break_min = max(0, BREAK_INTERVAL_MIN - elapsed_min)
            break_recommended = (remaining_break_min == 0)
            
            driving_stats = {
                "driving_time_str": driving_time_str,
                "remaining_break_min": remaining_break_min,
                "break_recommended": break_recommended
            }

            if results.multi_face_landmarks:
                landmarks_list = landmarks_to_points(results.multi_face_landmarks[0], w, h)
                ear = avg_ear(landmarks_list)
                mar = mar_from_landmarks(landmarks_list)
                
                # Head pose
                pitch, yaw, roll = get_head_pose(results.multi_face_landmarks[0].landmark, w, h)
                
                # Seatbelt Detection (Run every 5 frames)
                seatbelt_frame_count += 1
                if seatbelt_frame_count % 5 == 0:
                    is_seatbelt_detected = seatbelt_detector.detect(frame, results.multi_face_landmarks[0], w, h)
                    
                    # Hysteresis Logic
                    if is_seatbelt_detected:
                        seatbelt_worn_frames += 1
                        seatbelt_not_worn_frames = 0
                    else:
                        seatbelt_not_worn_frames += 1
                        seatbelt_worn_frames = 0
                    
                    # Require 3 consecutive checks (approx 15 frames / 0.5-1s) to change status
                    if seatbelt_worn_frames >= 3:
                        seatbelt_status = "WORN"
                    elif seatbelt_not_worn_frames >= 3:
                        seatbelt_status = "NOT_WORN"

                # Nodding (Pitch) - Drowsiness
                # Increased threshold to avoid false positives when looking down slightly
                if is_head_nodding(pitch, threshold_down=25): 
                    nodding_frames += 1
                else:
                    nodding_frames = 0

                # Turning (Yaw) - Distraction (Side Mirrors)
                # Check if head is turned significantly (e.g. > 20 degrees)
                # Allow for 4 seconds (approx 80-100 frames) before alerting
                if abs(yaw) > 20:
                    turning_frames += 1
                else:
                    turning_frames = 0

                # EAR logic
                if ear < EAR_THRESH:
                    closed_frames += 1
                else:
                    closed_frames = 0

                # MAR logic (yawning)
                if mar > MAR_THRESH:
                    yawning_frames += 1
                else:
                    yawning_frames = 0

                # Decide drowsy / distracted
                is_drowsy = False
                reason = None
                
                # Drowsiness Conditions
                # Condition 1: Eyes Closed (Only if NO sunglasses)
                if closed_frames >= EAR_CONSEC_FRAMES and not is_sunglasses_detected:
                    is_drowsy = True
                    reason = f"eyes_closed ({closed_frames}) ear={ear:.2f}"
                
                # Condition 2: Yawning (Always check)
                elif yawning_frames >= MAR_CONSEC_FRAMES:
                    is_drowsy = True
                    reason = f"yawning ({yawning_frames}) mar={mar:.2f}"
                
                # Condition 3: Head Nodding (Always check)
                elif nodding_frames >= 20:  # Reduced to ~1 sec for easier testing
                    is_drowsy = True
                    reason = f"head_nodding ({nodding_frames}) pitch={pitch:.2f}"
                
                # Debug prints for tuning
                if nodding_frames > 0:
                    print(f"Nodding Frames: {nodding_frames} (Pitch: {pitch:.1f})")
                elif sunglasses_frame_count % 30 == 0: # Print pitch every ~1 sec to help user debug
                    print(f"Current Pitch: {pitch:.1f} (Threshold: 20.0)")
                
                # Distraction Condition (treated as drowsy/alert for now, or separate status)
                # User said "it should buzz", so we can treat it as an alert condition.
                elif turning_frames >= 80: # ~4 seconds at 20 FPS
                    is_drowsy = True # Trigger alarm
                    reason = f"distracted_turning ({turning_frames}) yaw={yaw:.2f}"

                # Calculate Scores for Live Dashboard
                # Eye Score: 100 (Open) -> 0 (Closed)
                # Simple mapping: if ear > 0.3 -> 100, if ear < 0.15 -> 0
                eye_score = max(0, min(100, int((ear - 0.15) / (0.3 - 0.15) * 100)))
                
                # Yawn Score: 0 (Closed) -> 100 (Open)
                yawn_score = max(0, min(100, int((mar - 0.1) / (0.6 - 0.1) * 100)))
                
                # Tilt Score: 0 (Upright) -> 100 (Down)
                tilt_score = max(0, min(100, int((pitch - 10) / (40 - 10) * 100)))
                
                # Overall Risk
                _, is_using_phone, detected_phones = phone_detector.detect(frame)
                
                # Sunglasses Detection (Run every 30 frames)
                sunglasses_frame_count += 1
                if sunglasses_frame_count % 30 == 0:
                    is_sunglasses_detected = sunglasses_detector.detect(frame, results.multi_face_landmarks[0], w, h)
                
                if is_using_phone:
                    status_text = "PHONE USAGE DETECTED!"
                    color = (0, 0, 255)
                    overall_risk = 100 # Max risk
                    phone_status = "DETECTED"
                    
                    # Alert Logic
                    # Check cooldown
                    if time.time() - last_alert_time > 10: # 10s cooldown
                        last_alert_time = time.time()
                        
                        # TTS Alert
                        from alarm import play_tts_alert, buzz_for_duration
                        import threading
                        
                        # Run alerts in thread to not block video
                        def alert_sequence():
                            play_tts_alert("Please do not use mobile phone while driving")
                            buzz_for_duration(2)
                            
                        threading.Thread(target=alert_sequence, daemon=True).start()
                        
                        send_alert(driver=DRIVER_NAME, status="Phone Usage", extra={"reason": "phone_near_face_or_hand"})
                else:
                    phone_status = "NOT_DETECTED"

                # Draw Phone Boxes
                for ph in detected_phones:
                    x1, y1, x2, y2 = map(int, ph["box"])
                    # Red box if in use, else Yellow
                    box_color = (0, 0, 255) if is_using_phone else (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                # --- Sunglasses Alert Logic ---
                if is_sunglasses_detected:
                    sunglasses_status = "DETECTED"
                    if not is_using_phone: # Priority to phone
                        status_text = "SUNGLASSES DETECTED!"
                        color = (0, 0, 255)
                    overall_risk = max(overall_risk, 60) # Moderate risk (impairs eye tracking)
                    
                    # Alert every 10 seconds (frequent reminder)
                    if time.time() - last_sunglasses_alert_time > 10:
                        last_sunglasses_alert_time = time.time()
                        
                        from alarm import play_tts_alert, buzz_for_duration
                        import threading
                        
                        def sunglasses_alert():
                            play_tts_alert("Please remove sunglasses")
                            buzz_for_duration(1) # Beep for 1 second
                            
                        threading.Thread(target=sunglasses_alert, daemon=True).start()
                else:
                    sunglasses_status = "NOT_DETECTED"

                # --- Seatbelt Alert Logic ---
                if seatbelt_status == "NOT_WORN":
                    if not is_using_phone and not is_sunglasses_detected:
                        status_text = "SEATBELT NOT WORN!"
                        color = (0, 0, 255)
                    overall_risk = max(overall_risk, 80) # High risk
                    
                    # Alert every 3 minutes (180 seconds)
                    if time.time() - last_seatbelt_alert_time > 180:
                        last_seatbelt_alert_time = time.time()
                        
                        from alarm import play_tts_alert
                        import threading
                        
                        def seatbelt_alert():
                            play_tts_alert("Please wear your seatbelt")
                            
                        threading.Thread(target=seatbelt_alert, daemon=True).start()
                        send_alert(driver=DRIVER_NAME, status="Seatbelt", extra={"reason": "seatbelt_not_worn"})

                # --- Smoking Detection (ENABLED) ---
                smoking_frame_count += 1
                detected_smoking_objects = []
                
                # Process Hands
                hand_results = hands.process(rgb)
                
                if smoking_frame_count % 5 == 0:
                    is_smoking, detected_smoking_objects = smoking_detector.detect(
                        frame, 
                        results.multi_face_landmarks[0], 
                        w, h, 
                        hand_landmarks_list=hand_results.multi_hand_landmarks
                    )
                    if is_smoking:
                        smoking_status = "DETECTED"
                        overall_risk += 20
                    else:
                        smoking_status = "NOT_DETECTED"
                
                if smoking_status == "DETECTED":
                     if not is_using_phone:
                        status_text = "SMOKING DETECTED!"
                        color = (0, 0, 255)
                
                for box in detected_smoking_objects:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Smoking", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # --- Alarm Logic ---
                # Trigger alarm ONCE when drowsiness is detected
                
                if is_drowsy:
                    status_text = "DROWSY - ALARM!"
                    color = (0,0,255)
                    
                    # Play alarm once per episode
                    if not alarm_has_played:
                        play_alarm()
                        alarm_has_played = True
                    
                    # Send alert to backend (throttle to once every 5s to avoid spamming logs)
                    if time.time() - last_alert_time > 5:
                        last_alert_time = time.time()
                        send_alert(driver=DRIVER_NAME, status="Drowsy", extra={"reason": reason, "ear": round(ear,3), "mar": round(mar,3)})
                else:
                    # Reset alarm flag when condition clears
                    alarm_has_played = False
                    
                    # update status text with EAR/MAR for demo if not other critical status
                    if status_text == "Active":
                         status_text = f"EAR:{ear:.2f} MAR:{mar:.2f}"

                # draw face landmarks (optional)
                for (x,y) in landmarks_list:
                    cv2.circle(frame, (x,y), 1, (0,255,0), -1)

                cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Display Seatbelt Status
                sb_color = (0, 255, 0) if seatbelt_status == "WORN" else (0, 0, 255)
                cv2.putText(frame, f"Seatbelt: {seatbelt_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sb_color, 2)
                
                # --- DEBUG OVERLAY ---
                # Show Sunglasses Stats
                # We need to access the last calculated stats from the detector if possible, 
                # or we can just print the status for now. 
                # Ideally, we'd modify the detector to return stats, but for now let's show what we have.
                cv2.putText(frame, f"SunG: {sunglasses_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Show Smoking Stats
                cv2.putText(frame, f"Smoke: {smoking_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                
                # Show Head Pose
                cv2.putText(frame, f"Pitch: {int(pitch)} Yaw: {int(yaw)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                # Write Live Data to JSON for Dashboard
                write_live_data(eye_score, yawn_score, tilt_score, overall_risk, driving_stats, seatbelt_status, phone_status, sunglasses_status, smoking_status)

            else:
                cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)
                closed_frames = 0
                yawning_frames = 0
                write_live_data(0, 0, 0, 0, driving_stats) # Reset if no face but keep stats
                
                # Stop alarm if face lost (safety choice, or keep buzzing? usually stop to avoid annoyance if driver leaves)
                alarm_has_played = False

            # Show frame
            cv2.imshow("Driver Monitor (press q to quit, s to toggle seatbelt)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if seatbelt_status == "WORN":
                    seatbelt_status = "NOT_WORN"
                    print("Seatbelt: NOT_WORN (Manual)")
                else:
                    seatbelt_status = "WORN"
                    print("Seatbelt: WORN (Manual)")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
