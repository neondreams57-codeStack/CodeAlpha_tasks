"""
SORT (Simple Online and Realtime Tracking) Algorithm Implementation
Based on: https://arxiv.org/abs/1602.00763

Uses Kalman Filter for state estimation + Hungarian Algorithm for data association.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

#  Kalman Filter for bounding box state estimation

class KalmanBoxTracker:
    """
    Tracks a single bounding box using a Kalman Filter.
    State vector: [x_center, y_center, scale, aspect_ratio, vx, vy, vs]
    """
    count = 0  # class-level ID counter

    def __init__(self, bbox: np.ndarray):
        """
        Initialize a tracker from an initial bounding box [x1, y1, x2, y2].
        """
        # Kalman Filter matrices (constant velocity model)
        dt = 1.0  # time step

        # State transition matrix (7x7)
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0],
            [0, 1, 0, 0, 0,  dt, 0],
            [0, 0, 1, 0, 0,  0,  dt],
            [0, 0, 0, 1, 0,  0,  0],
            [0, 0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0, 0,  0,  1],
        ], dtype=float)

        # Measurement matrix (4x7): observe [x, y, s, r]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Measurement noise covariance
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

        # Process noise covariance
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])

        # State covariance (initial uncertainty)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0])

        # State vector
        self.x = np.zeros((7, 1))
        self.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: list = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    # Kalman operations 

    def predict(self) -> np.ndarray:
        """Advance state estimate and return predicted bounding box."""
        # Clamp scale to prevent negative values
        if (self.x[6] + self.x[2]) <= 0:
            self.x[6] = 0.0

        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        pred = self._z_to_bbox(self.x[:4])
        self.history.append(pred)
        return pred

    def update(self, bbox: np.ndarray) -> None:
        """Update state with a new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Return the current bounding box [x1, y1, x2, y2]."""
        return self._z_to_bbox(self.x[:4])

    # Coordinate helpers

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """[x1,y1,x2,y2] → [cx, cy, s, r]  (center, scale, aspect ratio)."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w * h          # area as scale
        r = w / float(h) if h > 0 else 1.0
        return np.array([[cx], [cy], [s], [r]], dtype=float)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """[cx, cy, s, r] → [x1, y1, x2, y2]."""
        w = np.sqrt(abs(z[2] * z[3]))
        h = abs(z[2]) / w if w > 0 else 1.0
        return np.array([
            z[0] - w / 2.0,
            z[1] - h / 2.0,
            z[0] + w / 2.0,
            z[1] + h / 2.0,
        ], dtype=float).flatten()


#  IoU helper

def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Compute IoU between all pairs of boxes.
    bb_test: (N, 4), bb_gt: (M, 4)  → returns (N, M) matrix.
    """
    bb_gt = bb_gt[np.newaxis, :, :]   # (1, M, 4)
    bb_test = bb_test[:, np.newaxis, :]  # (N, 1, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[...,  2] - bb_gt[...,  0]) * (bb_gt[...,  3] - bb_gt[...,  1])
    union = area_test + area_gt - intersection

    return intersection / np.maximum(union, 1e-6)


#  Hungarian data association

def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, list, list]:
    """
    Match detections to existing trackers using Hungarian algorithm.

    Returns:
        matches       : (K, 2) array of [det_idx, trk_idx] pairs
        unmatched_det : list of unmatched detection indices
        unmatched_trk : list of unmatched tracker indices
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(len(detections))),
            [],
        )

    iou_mat = iou_batch(detections, trackers)

    if min(iou_mat.shape) > 0:
        row_idx, col_idx = linear_sum_assignment(-iou_mat)
        matched_indices = np.stack([row_idx, col_idx], axis=1)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_det = [d for d in range(len(detections))
                     if d not in matched_indices[:, 0]]
    unmatched_trk = [t for t in range(len(trackers))
                     if t not in matched_indices[:, 1]]

    # Discard low-IoU matches
    matches = []
    for m in matched_indices:
        if iou_mat[m[0], m[1]] < iou_threshold:
            unmatched_det.append(m[0])
            unmatched_trk.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches_arr = (
        np.concatenate(matches, axis=0)
        if matches
        else np.empty((0, 2), dtype=int)
    )
    return matches_arr, unmatched_det, unmatched_trk

#  SORT Tracker

class Sort:
    """
    SORT: Simple Online and Realtime Tracker.

    Args:
        max_age        : Frames to keep a track alive without a detection.
        min_hits       : Minimum hits before a track is confirmed.
        iou_threshold  : Minimum IoU for a valid match.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def reset(self) -> None:
        """Reset tracker state and ID counter."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            dets: (N, 5) array of [x1, y1, x2, y2, score].
                  Pass np.empty((0,5)) for a frame with no detections.

        Returns:
            (M, 5) array of active tracks: [x1, y1, x2, y2, track_id].
        """
        self.frame_count += 1

        #Step 1: Predict new locations of existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = [*pos, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove trackers with NaN predictions
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = trks[~np.isnan(trks).any(axis=1)]

        #Step 2: Associate detections to trackers
        matched, unmatched_det, unmatched_trk = associate_detections_to_trackers(
            dets[:, :4], trks[:, :4], self.iou_threshold
        )

        #Step 3: Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        #Step 4: Create new trackers for unmatched detections
        for i in unmatched_det:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        #Step 5: Return confirmed tracks
        results = []
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                bbox = trk.get_state()
                results.append([*bbox, trk.id + 1])  # 1-indexed ID

        #Step 6: Prune dead trackers
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        return np.array(results) if results else np.empty((0, 5))
