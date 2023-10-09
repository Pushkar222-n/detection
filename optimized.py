import cv2
import time
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


if __name__ == '__main__':

    model = YOLO("box.pt")
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print("[ERROR] Cannot access webcam stream.")
        exit(0)
    # FPS of the input stream
    # Get the 5th property of VideoCapture object: cv2.CAP_PROP_POS_MSEC
    fps_input_stream = int(cap.get(5))
    print(f"FPS: {fps_input_stream}")

    success, frame = cap.read()  # reading single frame for hardware warm-up

    # delay variable is used in the code for simulating time taken for processing the frame.
    # Different amounts of delay can be used to evaluate performance.
    num_frames_processed = 0
    start = time.time()

    while True:
        success, frame = cap.read()
        if success is False:
            print("[EXIT].. No more frames to read")
            break

        # adding a delay for simulating video processing time
        delay = 0.03
        time.sleep(delay)
        num_frames_processed += 1

        results = model.predict(frame, imgsz=320, conf=0.8)
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            # for box in boxes:                                          # iterate boxes
            #     # get corner points as int
            #     r = box.xyxy[0]
            #     # print boxes
            #     print(r)
            #     cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)
            masks = result.masks
            frame = cv2.bitwise_and(frame, frame, mask=masks)

            probs = result.probs
            print(masks)
            print(probs)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    end = time.time()

    elapsed_time = end - start
    fps = num_frames_processed / elapsed_time
    print(f"FPS: {fps}, Elapsed Time: {elapsed_time}")

    cap.release()
    cv2.destroyAllWindows()
