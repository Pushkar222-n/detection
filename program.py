from ultralytics import YOLO
import cv2

model = YOLO("box.pt")
model.track(source="1", show=True, conf=0.6)
