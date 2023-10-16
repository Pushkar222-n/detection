import cv2
import time
import supervision as sv
from threading import Thread
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Don't forget to install supervision

# class WebCamStream:
#     def __init__(self, stream_id=0):
#         self.model = YOLO('box.pt')
#         self.stream_id = stream_id  # This is source, default is 0 for main camera

#         self.cap = cv2.VideoCapture(self.stream_id)

#         if not self.cap.isOpened():
#             print("[ERROR] Cannot access webcam stream.")
#             exit(0)
#         # Get the 5th property of VideoCapture object: cv2.CAP_PROP_POS_MSEC
#         fps_input_stream = int(self.cap.get(5))
#         print(f"FPS: {fps_input_stream}")

#         self.success, self.frame = self.cap.read()
#         if self.success is False:
#             print("[EXIT].. No more frames to read")
#             exit(0)

#         # self.stopped is initialized to False
#         self.stopped = True

#         # thread instatiation
#         # self.update runs in background to read the next available frame
#         self.t = Thread(target=self.update, args=())
#         self.t.daemon = True  # daemon threads handles background tasks

#     # Start thread
#     def start(self):
#         self.stopped = False
#         self.t.start()  # starts the thread activity

#     def update(self):
#         while True:
#             if self.stopped is True:
#                 break
#             self.success, self.frame = self.cap.read()
#             results = self.model.predict(self.frame, imgsz=320, conf=0.8)
#             for r in results:
#                 annotator = Annotator(self.frame)

#                 boxes = r.boxes
#                 for box in boxes:

#                     # get box coordinates in (top, left, bottom, right) format
#                     b = box.xyxy[0]
#                     c = box.cls
#                     annotator.box_label(b, self.model.names[int(c)])

#             self.frame = annotator.result()
#             if self.success is False:
#                 print("[EXIT].. No more frames to read")
#                 self.stopped = True
#                 break
#         self.cap.release()

#     # method to return latest frame
#     def read(self):
#         return self.frame

#     # method to stop reading framess
#     def stop(self):
#         self.stopped = True


# if __name__ == '__main__':
#     webcam_stream = WebCamStream(stream_id=0)
#     webcam_stream.start()

#     num_frames_processed = 0
#     start = time.time()
#     while True:
#         if webcam_stream.stopped is True:
#             break
#         else:
#             frame = webcam_stream.read()

#         # adding a delay for simulating video processing time
#         delay = 0.03
#         time.sleep(delay)
#         num_frames_processed += 1

#         cv2.imshow('frame', frame)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#     end = time.time()
#     webcam_stream.stop()

#     elapsed_time = end - start
#     fps = num_frames_processed / elapsed_time
#     print(f"FPS: {fps}, Elapsed Time: {elapsed_time}")

#     cv2.destroyAllWindows()


def main():

    model = YOLO("box.pt")

    # box_annotator = sv.BoxAnnotator(
    #     thickness=2,
    #     text_thickness=2,
    #     text_scale=1
    # )
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam stream.")
        exit(0)

    # FPS of the input stream
    # Get the 5th property of VideoCapture object: cv2.CAP_PROP_POS_MSEC
    fps_input_stream = int(cap.get(5))
    print(f"FPS: {fps_input_stream}")
    for _ in range(5):
        success, frame = cap.read()  # reading single frame for hardware warm-up

    # delay variable is used in the code for simulating time taken for processing the frame.
    # Different amounts of delay can be used to evaluate performance.
    num_frames_processed = 0
    start = time.time()

    while True:
        success, frame = cap.read()
        H, W, channels = frame.shape
        if not success:
            print("[EXIT].. No more frames to read")
            break

        # adding a delay for simulating video processing time
        delay = 0.03
        time.sleep(delay)
        num_frames_processed += 1

        results = model.predict(
            frame, stream=True, imgsz=640, conf=0.7, agnostic_nms=True)
        # # detections = sv.Detections.from_yolov8(results)
        count = 0
        for result in results:
            count += 1
            boxes = result.boxes  # Boxes object for bbox outputs
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # iterate boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.circle(frame, (W // 2, H // 2), 2, (0, 0, 255), 2)
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2),
                           2, (0, 255, 0), 2)

            # masks = result.masks
            # frame = cv2.bitwise_and(frame, frame, mask=masks)

            # probs = result.probs
        #     # print(masks)
        #     # print(probs)

        cv2.putText(frame, str(count), (0, 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
        cv2.imshow("Frame", mat=frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    end = time.time()

    elapsed_time = end - start
    fps = num_frames_processed / elapsed_time
    print(f"FPS: {fps}, Elapsed Time: {elapsed_time}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
