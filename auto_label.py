import cv2
import os
import glob
import mediapipe as mp
import numpy as np
from detection.smoking_detector import SmokingDetector

# Setup
DATASET_DIR = "datasets"
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")
os.makedirs(LABELS_DIR, exist_ok=True)

# Initialize Detector
detector = SmokingDetector()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def main():
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    print(f"Found {len(image_files)} images to label.")

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            # Run Detection (Custom Lenient Logic)
            # is_detected, detected_objects = detector.detect(frame, results.multi_face_landmarks[0], w, h, debug=True)
            
            # Custom Lenient Logic for Auto-Labeling
            mouth_indices = [61, 291, 0, 17]
            mouth_pts = [(results.multi_face_landmarks[0].landmark[i].x * w, results.multi_face_landmarks[0].landmark[i].y * h) for i in mouth_indices]
            mouth_center = np.mean(mouth_pts, axis=0)
            
            # Use YOLO directly
            yolo_results = detector.model(frame, verbose=False, conf=0.1) # Lower confidence
            
            best_box = None
            min_dist = float('inf')
            
            for r in yolo_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls != 79: continue # Toothbrush
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    obj_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    
                    # Distance check (Relaxed)
                    dist = np.linalg.norm(obj_center - mouth_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_box = box
            
            # If we found ANY toothbrush near the mouth, take it.
            if best_box is not None and min_dist < 200: # Relaxed distance
                is_detected = True
                detected_objects = [best_box]
            else:
                is_detected = False
                detected_objects = []
            
            if is_detected and detected_objects:
                # Use the first detected object
                box = detected_objects[0]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Convert to Normalized YOLO format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Class 0 = cigarette
                label_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                
                # Save Label
                filename = os.path.basename(img_path).replace(".jpg", ".txt")
                label_path = os.path.join(LABELS_DIR, filename)
                
                with open(label_path, "w") as f:
                    f.write(label_line)
                
                print(f"Auto-labeled: {filename}")
            else:
                print(f"No cigarette detected in: {os.path.basename(img_path)}")
        else:
            print(f"No face found in: {os.path.basename(img_path)}")

if __name__ == "__main__":
    main()
