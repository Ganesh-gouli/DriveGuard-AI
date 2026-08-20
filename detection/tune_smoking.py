import cv2
import mediapipe as mp
import numpy as np
import os
from smoking_detector import SmokingDetector

# --- Setup MediaPipe (IMPORTANT: static_image_mode=False for accuracy) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Initialize Smoking Detector
detector = SmokingDetector()

# Image directory
base_path = os.path.dirname(os.path.abspath(__file__))

images = [
    ("Smoking 1", "debug_smoking_Smoking_3.jpg"),
    ("Smoking 2", "debug_smoking_Smoking_4.jpg"),
]

print(f"{'Image':<25} | {'Result'}")
print("-" * 40)

for label, filename in images:

    # Load Image
    path = os.path.join(base_path, filename)
    frame = cv2.imread(path)

    if frame is None:
        print(f"{label:<25} | ERROR: Cannot read {filename}")
        continue

    # YOLO works better on slightly larger images → Resize
    frame = cv2.resize(frame, (960, 540))

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe landmarks
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print(f"{label:<25} | No Face Detected")
        continue

    landmarks = results.multi_face_landmarks[0]

    # --- Run Smoking Detection with debug data ---
    try:
        is_detected, detected_objects = detector.detect(
            frame, landmarks, w, h, debug=True
        )
    except:
        is_detected = detector.detect(frame, landmarks, w, h)
        detected_objects = []

    result_str = "DETECTED" if is_detected else "NOT DETECTED"
    print(f"{label:<25} | {result_str}")

    # --- Draw Mouth ROI ---
    mouth_idxs = [61, 291, 0, 17]
    mouth_points = [
        (int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
        for i in mouth_idxs
    ]

    mouth_center = np.mean(mouth_points, axis=0).astype(int)
    mouth_width = int(np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[1])))
    proximity_radius = int(mouth_width * 3.0)

    cv2.circle(frame, tuple(mouth_center), proximity_radius, (0, 0, 255), 2)

    # --- Draw YOLO Boxes ---
    for box in detected_objects:
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        conf = float(box.conf[0])
        cls_name = detector.model.names[int(box.cls[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{cls_name} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # --- Save Output ---
    output_filename = f"verified_yolo_smoking_{label.replace(' ', '_')}.jpg"
    cv2.imwrite(output_filename, frame)
    print(f"Saved visualization to {output_filename}\n")
