import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
model1 = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Load the pre-trained RetinaNet model
model2 = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Set the threshold for detecting objects
conf_threshold = 0.5

# Open a connection to the webcam feed
cap = cv2.VideoCapture('http://38.81.159.248/mjpg/video.mjpg')

# Define the class names for the RetinaNet model
class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Create two windows
cv2.namedWindow('MobileNet SSD', cv2.WINDOW_NORMAL)
cv2.namedWindow('RetinaNet', cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the webcam feed
    ret, frame = cap.read()

    # Apply MobileNet SSD object detection to the frame and display it in the first window
    blob1 = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    model1.setInput(blob1)
    detections1 = model1.forward()

    for i in range(detections1.shape[2]):
        confidence = detections1[0, 0, i, 2]
        if confidence > conf_threshold:
            class_id = int(detections1[0, 0, i, 1])
            class_name = class_names[class_id]
            box = detections1[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x, y, w, h = box.astype('int')
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('MobileNet SSD', frame)

    # Apply RetinaNet object detection to the frame and display it in the second window
    blob2 = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    model2.setInput(blob2)
    detections2 = model2.forward()

    for i in range(detections1.shape[2]):
        confidence = detections2[0, 0, i, 2]
        if confidence > conf_threshold:
            class_id = int(detections2[0, 0, i, 1])
            class_name = class_names[class_id]
            box = detections2[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            x, y, w, h = box.astype('int')
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('RetinaNet', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources used by the webcam feed
cap.release()

# Close all windows
cv2.destroyAllWindows()
