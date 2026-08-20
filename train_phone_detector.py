from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data='{{DATASET_PATH}}/data.yaml',  # path to dataset YAML
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu', # or 0 for GPU
    project='driver_phone_detection',
    name='yolov8n_phone_v1',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=1e-3,
    augment=True,
    val=True,
    patience=10,
    save=True
)

# Export the model
success = model.export(format='onnx')
