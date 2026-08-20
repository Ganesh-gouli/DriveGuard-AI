import cv2
import os
import glob

# Configuration
DATASET_DIR = "datasets"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Classes
CLASSES = ["cigarette", "phone", "sunglasses"]
CURRENT_CLASS_ID = 0 # Default to cigarette

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_img, temp_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = current_img.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(current_img, (ix, iy), (x, y), (0, 255, 0), 2)
        temp_img = current_img.copy()
        
        # Save Label (Normalized YOLO format)
        h, w, _ = current_img.shape
        x_center = ((ix + x) / 2) / w
        y_center = ((iy + y) / 2) / h
        width = abs(x - ix) / w
        height = abs(y - iy) / h
        
        label_line = f"{CURRENT_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        # Determine if train or val
        img_path = param
        rel_path = os.path.relpath(img_path, IMAGES_DIR)
        label_path = os.path.join(LABELS_DIR, rel_path.replace(".jpg", ".txt"))
        
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, "a") as f:
            f.write(label_line)
        print(f"Saved Label: {CLASSES[CURRENT_CLASS_ID]} -> {label_path}")

drawing = False
ix, iy = -1, -1
current_img = None
temp_img = None

def main():
    global current_img, temp_img, CURRENT_CLASS_ID
    
    # Find all images
    image_files = glob.glob(os.path.join(IMAGES_DIR, "**", "*.jpg"), recursive=True)
    
    if not image_files:
        print("No images found in datasets/images. Run collect_data.py first!")
        return

    print("--------------------------------------------------")
    print("LABELING TOOL")
    print("--------------------------------------------------")
    print("Draw box with mouse.")
    print("Press 'c' to switch class (Current: cigarette)")
    print("Press 'n' for next image")
    print("Press 'q' to quit")
    print("--------------------------------------------------")

    for img_path in image_files:
        print(f"Labeling: {img_path}")
        current_img = cv2.imread(img_path)
        if current_img is None:
            continue
        temp_img = current_img.copy()
        
        cv2.namedWindow("Labeler")
        cv2.setMouseCallback("Labeler", mouse_callback, param=img_path)

        while True:
            cv2.imshow("Labeler", temp_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                return
            elif key == ord('n'):
                break
            elif key == ord('c'):
                CURRENT_CLASS_ID = (CURRENT_CLASS_ID + 1) % len(CLASSES)
                print(f"Switched Class to: {CLASSES[CURRENT_CLASS_ID]}")

    cv2.destroyAllWindows()
    print("All images processed.")

if __name__ == "__main__":
    main()
