import numpy as np
from scipy.optimize import linear_sum_assignment
from src.kalman import KalmanBoxTracker

def iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-6)

class MultiObjectTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """detections: list[bbox]"""
        for t in self.trackers:
            t.predict()

        if len(self.trackers) == 0:
            for d in detections:
                self.trackers.append(KalmanBoxTracker(d))
            return []

        trk_boxes = [t.get_state() for t in self.trackers]
        iou_matrix = np.zeros((len(detections), len(trk_boxes)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trk_boxes):
                iou_matrix[d, t] = iou(det, trk)

        rows, cols = linear_sum_assignment(-iou_matrix)
        matched, unmatched_dets = [], list(range(len(detections)))
        for r, c in zip(rows, cols):
            if iou_matrix[r, c] < self.iou_threshold:
                continue
            self.trackers[c].update(detections[r])
            matched.append((r, c))
            if r in unmatched_dets:
                unmatched_dets.remove(r)

        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d]))

        results = []
        for t in list(self.trackers):
            if t.time_since_update > self.max_age:
                self.trackers.remove(t)
            elif t.hits >= self.min_hits or t.time_since_update == 0:
                results.append({"id": t.id, "bbox": t.get_state()})
        return results
