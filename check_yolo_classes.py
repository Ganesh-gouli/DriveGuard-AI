from ultralytics import YOLO
import os

try:
    model = YOLO("yolov8n.pt")
    print("Model Classes:")
    print(model.names)
except Exception as e:
    print(f"Error loading model: {e}")
