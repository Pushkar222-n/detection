from ultralytics import YOLO
import cv2

model = YOLO("box.pt")
model.predict(source="1", show=True, conf=0.6)
