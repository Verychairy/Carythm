import cv2
import torch
import numpy as np

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # Nano version for speed

# Connect to CCTV feed
cap = cv2.VideoCapture("rtsp://your_cctv_stream_url")
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect cars using YOLO
    results = model(frame)
    detections = results.xyxy[0].numpy()  # [x1, y1, x2, y2, confidence, class]

    # Process detections
    for x1, y1, x2, y2, conf, cls in detections:
        if cls == 2:  # Class 2 is "car" in YOLO's COCO dataset
            # Filter by motion direction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion_direction = np.mean(flow[..., 0])  # Horizontal motion
                if motion_direction > 0:  # Rightward motion
                    print("Car detected moving right!")

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display frame
    cv2.imshow("Traffic Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
