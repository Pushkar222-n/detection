import cv2
import time
from threading import Thread
from ultralytics import YOLO
from utils import get_params

SOURCE = 1

cap = cv2.VideoCapture(SOURCE)
model = YOLO("box.pt")
while True:
    success, frame = cap.read()
    if success:
        height, width, channels = frame.shape
        results = model.predict(frame, agnostic_nms=True)
        bboxes, class_ids, segmentations, scores = get_params(
            height, width, results)
        for bbox, class_id, seg, score in zip(bboxes, class_ids, scores):
            (x, y, x2, y2) = bbox
            print(f"Box at: ({(x + x2)/ 2},{(y + y2)/2}) ")

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
            # cv2.polylines(frame, [seg], True, (255, 0, 0), 2)

            cv2.putText(frame, str(class_id), (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

        cv2.imshow("FRAME", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

    cv2.destroyAllWindows()
