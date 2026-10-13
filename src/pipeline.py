import cv2
from ultralytics import YOLO
from src.tracker import MultiObjectTracker

class TrackingPipeline:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.tracker = MultiObjectTracker()

    def process(self, frame):
        res = self.model.predict(frame, verbose=False)[0]
        dets = [b.xyxy[0].tolist() for b in res.boxes]
        tracks = self.tracker.update(dets)
        for t in tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {t['id']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame, tracks

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    p = TrackingPipeline()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frame, _ = p.process(frame)
        cv2.imshow("UAV Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
