from ultralytics import YOLO
import numpy as np
import cv2


# def get_params(frame_height, frame_width, results):
#     height, width = frame_height, frame_width
#     result = results[0]
#     segmentation_contours_index = []
#     for mask in result.masks:
#         for seg in mask:
#             seg[:, 0] *= width
#             seg[:, 1] *= height
#             segment = np.array(seg, dtype=np.int32)
#             segmentation_contours_index.append(segment)

#     bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
#     class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
#     scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
#     return bboxes, class_ids, segmentation_contours_index, scores

model = YOLO("box.pt")
SOURCE = 1
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam stream")

# Warmup
for _ in range(5):
    success, frame = cap.read()


while True:
    success, frame = cap.read()
    frame_height, frame_width, channels = frame.shape
    if not success:
        print("[EXIT].... No more frames to read")
        break
    results = model.predict(frame, imgsz=320, conf=0.8, agnostic_nms=True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # masks = result.masks
        # for j, mask in enumerate(result.masks.data):
        #     mask = mask.numpy() * 255
        #     polygon = mask.xy[0]
        #     mask = cv2.resize = (mask, (frame_width, frame_height))
    cv2.imshow(frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


# def params(results):
#     # for result in results:
#     #     for j, mask
