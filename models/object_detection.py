from ultralytics import YOLO
import cv2

# Load YOLOv8n pretrained model (small and fast)
yolo_model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = yolo_model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = yolo_model.names[cls]

        cv2.rectangle(frame, (x1, y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    return frame
