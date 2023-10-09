from ultralytics import YOLO
import cv2

model = YOLO("box.pt")
model.predict(source="2", show=True, conf=0.6)

# import cv2
# import ultralytics
# from ultralytics import YOLO


# def access_webcam_safely(source):
#     """
#     Safely accesses the webcam with the specified source.

#     Args:
#       source: The source of the webcam. This can be a device number or a file path.

#     Returns:
#       A cv2.VideoCapture object.
#     """

#     # Capture the video from the webcam
#     cap = cv2.VideoCapture(source)

#     # Check if the webcam is open
#     if not cap.isOpened():
#         print("Could not open webcam")
#         exit()

#     return cap


# def run_yolo(source, model):
#     """
#     Runs YOLOv5 on the video stream from the specified source.

#     Args:
#       source: The source of the webcam. This can be a device number or a file path.
#       model: The YOLOv5 model.
#     """

#     # Access the webcam safely
#     cap = access_webcam_safely(source)

#     # Loop over the video stream
#     while True:

#         # Read a frame from the webcam
#         ret, frame = cap.read()

#         # If the frame is not empty, run YOLOv5 on it
#         if ret:
#             model.predict(frame)

#             # Display the output video stream with the detected objects highlighted
#             cv2.imshow("Webcam", frame)

#         # Wait for a key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam
#     cap.release()


# # Run YOLOv5 on the video stream from webcam 1
# run_yolo(1, model)


# import cv2
# import threading
# import ultralytics
# from ultralytics import YOLO


# def access_webcam_safely(source):
#     """
#     Safely accesses the webcam with the specified source.

#     Args:
#       source: The source of the webcam. This can be a device number or a file path.

#     Returns:
#       A cv2.VideoCapture object.
#     """

#     # Capture the video from the webcam
#     cap = cv2.VideoCapture(source)

#     # Check if the webcam is open
#     if not cap.isOpened():
#         print("Could not open webcam")
#         exit()

#     return cap


# class camThread(threading.Thread):
#     def __init__(self, previewName, camID, model):
#         threading.Thread.__init__(self)
#         self.previewName = previewName
#         self.camID = camID
#         self.model = model

#     def run(self):
#         print("Starting ", self.previewName)
#         camPreview(self.previewName, self.camID, self.model)


# def camPreview(previewName, camID, model):
#     cv2.namedWindow(previewName)
#     cap = access_webcam_safely(camID)

#     while True:

#         # Read a frame from the webcam
#         ret, frame = cap.read()

#         # If the frame is not empty, run YOLOv5 on it
#         if ret:
#             model.predict(frame)

#             # Display the output video stream with the detected objects highlighted
#             cv2.imshow(previewName, frame)

#         # Wait for a key press
#         key = cv2.waitKey(20)
#         if key == 27:  # exit on ESC
#             break

#     cv2.destroyWindow(previewName)


# # Create two threads and pass in the YOLOv5 model to each thread
# # thread1 = camThread("Camera 1", 1, model)
# thread2 = camThread("Camera 2", 2, model)

# # Start the threads
# # thread1.start()
# thread2.start()
