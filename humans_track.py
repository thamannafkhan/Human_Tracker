from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.track(source="people.mp4", classes=[0], tracker="bytetrack.yaml", conf=0.4, save=True)
print("Tracking complete. Results saved to 'runs/track/people' folder.")