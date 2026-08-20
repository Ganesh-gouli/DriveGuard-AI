import cv2
import mediapipe as mp
import numpy as np
import os
from sunglasses_detector import SunglassesDetector

# -------------------------------
# MediaPipe Setup
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,           # Better accuracy
    max_num_faces=1,
    refine_landmarks=True
)

# Detector
detector = SunglassesDetector()

# -------------------------------
# Image Paths
# -------------------------------
base_path = r"C:\Users\santi\.gemini\antigravity\brain\3bbb9540-f6f0-4302-9c99-45d6fc165d27"
images = [
    ("No Sunglasses (False Pos)", "uploaded_image_1764344567794.png"),
    ("Sunglasses 1", "uploaded_image_0_1764345205231.jpg"),
    ("Sunglasses 2", "uploaded_image_1_1764345205231.jpg"),
    ("Sunglasses 3", "uploaded_image_2_1764345205231.jpg"),
    ("Sunglasses 4", "uploaded_image_3_1764345205231.jpg"),
]

print(f"{'Image':<25} | {'Ratio':<6} | {'Var':<6} | {'Skin':<6} | {'Result'}")
print("-" * 70)

for label, filename in images:

    path = os.path.join(base_path, filename)
    frame = cv2.imread(path)

    if frame is None:
        print(f"{label:<25} | ERROR loading {filename}")
        continue

    # -------------------------------
    # Resize for more consistent stats
    # -------------------------------
    frame = cv2.resize(frame, (960, 540))
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print(f"{label:<25} | NO FACE")
        with open("tuning_results.txt", "a") as f:
            f.write(f"DATA,{label},NO_FACE\n")
        continue

    landmarks = results.multi_face_landmarks[0]

    # -------------------------------
    # Compute ROI Brightness Stats
    # -------------------------------
    LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
    SKIN_IDXS = [168, 6, 197, 195]

    l_mean, l_std = detector.get_roi_stats(frame, landmarks, LEFT_EYE_IDXS, w, h)
    r_mean, r_std = detector.get_roi_stats(frame, landmarks, RIGHT_EYE_IDXS, w, h)
    s_mean, s_std = detector.get_roi_stats(frame, landmarks, SKIN_IDXS, w, h)

    avg_eye_bright = (l_mean + r_mean) / 2
    avg_eye_std    = (l_std + r_std) / 2

    # Avoid false positives under very dark lighting
    if s_mean < 5:
        ratio = 1  # treat as CLEAR (cannot detect sunglasses in dark)
    else:
        ratio = avg_eye_bright / s_mean

    # -------------------------------
    # Run Sunglasses Detection
    # -------------------------------
    is_detected = detector.detect(frame, landmarks, w, h)
    result_str = "DETECTED" if is_detected else "CLEAR"

    print(f"{label:<25} | {ratio:.2f}  | {avg_eye_std:.2f} | {s_mean:.1f}  | {result_str}")

    # Save Analysis Log
    with open("tuning_results.txt", "a") as f:
        f.write(f"DATA,{label},{ratio:.4f},{avg_eye_std:.4f},{s_mean:.1f},{result_str}\n")

    # -------------------------------
    # Optional Debug Visualization
    # -------------------------------
    for idx in LEFT_EYE_IDXS + RIGHT_EYE_IDXS + SKIN_IDXS:
        lm = landmarks.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    out_file = f"debug_sunglasses_{label.replace(' ', '_')}.jpg"
    cv2.imwrite(out_file, frame)
