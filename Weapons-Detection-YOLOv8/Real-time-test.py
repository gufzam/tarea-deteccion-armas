import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO("best.pt")  # Replace with the path to the best.pt model on your desktop

# Start capturing video from the laptop's camera (use 0 for the default webcam)
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Perform inference on the captured frame
    results = model(frame)  # Run detection on the frame

    # Annotate the frame with bounding boxes for the detected objects
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow("Object Detection (Gun Detection)", annotated_frame)

    # Press 'q' to exit the camera window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()