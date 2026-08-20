from ultralytics import YOLO
import os
import yaml

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

def create_data_yaml():
    data = {
        'path': DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'cigarette',
            1: 'phone',
            2: 'sunglasses'
        }
    }
    
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path

def main():
    print("Preparing Training...")
    yaml_path = create_data_yaml()
    
    # Load model
    model = YOLO('yolov8n.pt') 
    
    print("Starting Training...")
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        project='driver_custom_model',
        name='yolov8n_custom_v1',
        exist_ok=True
    )
    
    print("Training Complete!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
