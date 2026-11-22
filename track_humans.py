import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam. Try changing 0 to 1 in VideoCapture.")
    exit()
cv2.namedWindow("Live Human Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Human Tracking", cv2.WND_PROP_TOPMOST, 1)

print("Live feed started! Click the video window and press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from camera.")
        break
    results = model.track(frame, classes=[0], persist=True, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Live Human Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
