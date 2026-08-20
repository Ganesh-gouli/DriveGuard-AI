import cv2
import mediapipe as mp
import numpy as np
import os
from seatbelt_detector import SeatbeltDetector

# Setup MediaPipe (IMPORTANT: static_image_mode=False)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Initialize Seatbelt Detector
detector = SeatbeltDetector()

# Image directory
base_path = r"C:\Users\santi\.gemini\antigravity\brain\3bbb9540-f6f0-4302-9c99-45d6fc165d27"

images = [
    ("Seatbelt 1", "uploaded_image_0_1764346353685.jpg"),
    ("Seatbelt 2", "uploaded_image_1_1764346353685.jpg"),
    ("Seatbelt 3", "uploaded_image_2_1764346353685.jpg"),
    ("Seatbelt 4", "uploaded_image_3_1764346353685.jpg"),
]

print(f"{'Image':<25} | {'Result'}")
print("-" * 45)

# Output file
output_file = "tuning_results_seatbelt.txt"
open(output_file, "w").close()  # Clear file

for label, filename in images:
    path = os.path.join(base_path, filename)
    frame = cv2.imread(path)

    if frame is None:
        print(f"{label:<25} | ERROR: Cannot read file")
        continue

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe Face Processing
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
    else:
        landmarks = None
        print(f"{label:<25} | No Face - Running Full-Frame Logic")

    # Seatbelt Detection
    is_detected = detector.detect(frame, landmarks, w, h)
    result_str = "DETECTED (WORN)" if is_detected else "NOT DETECTED"

    print(f"{label:<25} | {result_str}")

    # Log output
    with open(output_file, "a") as f:
        f.write(f"{label},{result_str}\n")

    # SAVE DEBUG IMAGE
    debug_img = frame.copy()
    cv2.putText(
        debug_img, result_str, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if is_detected else (0, 0, 255), 2
    )

    cv2.rectangle(debug_img, (10, 10), (w - 10, h - 10),
                  (0, 255, 0) if is_detected else (0, 0, 255), 2)

    cv2.imwrite(f"debug_seatbelt_{label.replace(' ', '_')}.jpg", debug_img)
