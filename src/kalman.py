import numpy as np

class KalmanBoxTracker:
    """Простой SORT-подобный трекер с Kalman-фильтром по bbox."""
    count = 0

    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.x = np.zeros((7, 1))
        cx, cy, s, r = self._to_z(bbox)
        self.x[:4] = np.array([[cx], [cy], [s], [r]])
        self.P = np.eye(7) * 10
        self.F = np.eye(7)
        for i in range(4):
            self.F[i, i + 3] = 1
        self.H = np.eye(4, 7)
        self.R = np.eye(4) * 1.0
        self.Q = np.eye(7) * 0.01
        self.time_since_update = 0
        self.hits = 1

    def _to_z(self, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return x1 + w/2, y1 + h/2, w*h, w/max(h, 1e-6)

    def _to_bbox(self, x):
        cx, cy, s, r = x[:4].flatten()
        w = np.sqrt(max(s * r, 1e-6))
        h = s / max(w, 1e-6)
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] *= 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._to_bbox(self.x)

    def update(self, bbox):
        z = np.array(self._to_z(bbox)).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        return self._to_bbox(self.x)
