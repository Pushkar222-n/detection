import cv2
import time
from threading import Thread
from ultralytics import YOLO


class WebCamStream:
    def __init__(self, stream_id=0):
        self.model = YOLO('yolov5s.pt')  # Replace with your YOLO model path
        self.stream_id = stream_id  # This is the source, default is 0 for the main camera

        self.cap = cv2.VideoCapture(self.stream_id)

        if not self.cap.isOpened():
            print("[ERROR] Cannot access the webcam stream.")
            exit(0)

        # Initialize frame variables
        self.success, self.frame = self.cap.read()
        if self.success is False:
            print("[EXIT].. No more frames to read")
            exit(0)

        # Initialize the stopped flag
        self.stopped = True

        # Thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # Daemon threads handle background tasks

    # Start the thread
    def start(self):
        self.stopped = False
        self.t.start()  # Start the thread activity

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.success, self.frame = self.cap.read()

            # Perform inference and draw bounding boxes on the frame
            results = self.model(self.frame)

            for pred in results.pred[0]:
                x1, y1, x2, y2, confidence, class_id = pred.tolist()
                if confidence > 0.5:  # Adjust the confidence threshold as needed
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    label = self.model.names[int(class_id)]
                    cv2.rectangle(self.frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    cv2.putText(self.frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if self.success is False:
                print("[EXIT].. No more frames to read")
                self.stopped = True
                break

    # Method to return the latest frame
    def read(self):
        return self.frame

    # Method to stop reading frames
    def stop(self):
        self.stopped = True


if __name__ == '__main__':
    webcam_stream = WebCamStream(stream_id=0)  # Change the stream_id as needed
    webcam_stream.start()

    num_frames_processed = 0
    start = time.time()
    while True:
        if webcam_stream.stopped is True:
            break
        else:
            frame = webcam_stream.read()

        # Adding a delay for simulating video processing time
        delay = 0.03
        time.sleep(delay)
        num_frames_processed += 1

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    end = time.time()
    webcam_stream.stop()

    elapsed_time = end - start
    fps = num_frames_processed / elapsed_time
    print(f"FPS: {fps}, Elapsed Time: {elapsed_time}")

    cv2.destroyAllWindows()
