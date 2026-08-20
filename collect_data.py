import cv2
import os
import time

# Configuration
DATASET_DIR = "datasets"
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "images", "train")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "images", "val")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("--------------------------------------------------")
    print("DATA COLLECTION TOOL")
    print("--------------------------------------------------")
    print("Press 's' to save to TRAIN")
    print("Press 'v' to save to VAL")
    print("Press 'q' to quit")
    print("--------------------------------------------------")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time() * 1000)
            filename = f"img_{timestamp}.jpg"
            path = os.path.join(TRAIN_IMG_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"Saved TRAIN: {filename}")
            count += 1
        elif key == ord('v'):
            timestamp = int(time.time() * 1000)
            filename = f"img_{timestamp}.jpg"
            path = os.path.join(VAL_IMG_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"Saved VAL: {filename}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Collected {count} images.")

if __name__ == "__main__":
    main()
