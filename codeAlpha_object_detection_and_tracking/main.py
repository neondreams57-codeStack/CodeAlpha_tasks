import argparse
import time
import cv2
import numpy as np

# Local SORT implementation
from sort_tracker import Sort


# Colour palette – 40 visually distinct track colours

_PALETTE = [
    (255,  56,  56), (255, 157,  151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), (72,  249,  10), (146, 204,  23), (61,  219,  134),
    (26,  147,  52), (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,   69, 147), (100,  115, 255), (0,   24, 236), (132,  56, 255),
    (82,    0, 133), (203,  56, 255), (255,  149, 200), (255,  55, 199),
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
    (255, 100, 255), (100, 255, 255), (200, 150,  50), (50,  200, 150),
    (150,  50, 200), (200,  50, 150), (50,  150, 200), (150, 200,  50),
    (230, 120,  60), (60,  230, 120), (120,  60, 230), (230, 230,  60),
    (60,  230, 230), (230,  60, 230), (180, 180,  80), (80,  180, 180),
]

def track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[int(track_id) % len(_PALETTE)]


# Drawing utilities
def draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    label: str,
    conf: float | None = None,
) -> None:
    """Draw a rounded bounding box with a filled label badge."""
    thick = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

    text = f"{label}  {conf:.2f}" if conf is not None else label
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    # Badge background
    badge_y1 = max(y1 - th - baseline - 6, 0)
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 8, y1), color, -1)
    cv2.putText(
        frame, text,
        (x1 + 4, y1 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    fps: float,
    n_tracks: int,
    frame_num: int,
    source_label: str,
) -> None:
    """Heads-up display overlay."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 34), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    info = (
        f"  {source_label}   |   "
        f"Frame: {frame_num:05d}   |   "
        f"Tracks: {n_tracks}   |   "
        f"FPS: {fps:5.1f}"
    )
    cv2.putText(frame, info, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

    # Bottom-right model badge
    badge = "YOLOv8 + SORT"
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (w - bw - 12, h - bh - 10), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, badge, (w - bw - 6, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA)


# Detection helpers

def yolo_detections_to_sort(results, class_filter: list[int] | None) -> np.ndarray:
    """
    Convert ultralytics Results object → SORT input array (N, 5).
    Each row: [x1, y1, x2, y2, confidence]
    """
    dets = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 5))

    for box in boxes:
        cls_id = int(box.cls[0])
        if class_filter and cls_id not in class_filter:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        dets.append([x1, y1, x2, y2, conf])

    return np.array(dets, dtype=float) if dets else np.empty((0, 5))

# Main pipeline

def run(args: argparse.Namespace) -> None:
    # Load YOLO model 
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed.\n"
            "Run:  pip install ultralytics"
        )

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    class_names: dict[int, str] = model.names  # {0: 'person', 1: 'bicycle', ...}

    class_filter: list[int] | None = args.classes or None

    # Open video source 
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open source: {src}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_label = "Webcam" if src == 0 else str(args.source).split("/")[-1]

    print(f"[INFO] Source : {source_label}  ({frame_w}x{frame_h} @ {cam_fps:.1f} fps)")

    # Optional video writer 
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, cam_fps, (frame_w, frame_h))
        print(f"[INFO] Saving output → {args.save}")

    # Initialize SORT tracker 
    tracker = Sort(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
    )

    # Per-track confidence memory 
    # SORT doesn't carry confidence; keep last known conf per track ID
    track_conf_map: dict[int, float] = {}
    track_class_map: dict[int, str] = {}

    # FPS measurement
    fps_window = 30
    ts_history: list[float] = []
    frame_num = 0

    print("[INFO] Running  ·  Press Q to quit  ·  Press S to screenshot")
    print("─" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        frame_num += 1
        t0 = time.perf_counter()

        # Detection
        yolo_results = model(
            frame,
            conf=args.conf,
            iou=args.nms_iou,
            verbose=False,
            classes=class_filter,
        )
        dets = yolo_detections_to_sort(yolo_results, class_filter)

        # Build a detection-to-class/conf lookup (index → class, conf)
        det_meta: dict[int, tuple[str, float]] = {}
        boxes = yolo_results[0].boxes
        if boxes is not None and len(boxes):
            idx = 0
            for box in boxes:
                cls_id = int(box.cls[0])
                if class_filter and cls_id not in class_filter:
                    continue
                det_meta[idx] = (class_names.get(cls_id, str(cls_id)), float(box.conf[0]))
                idx += 1

        # Tracking
        tracks = tracker.update(dets)

        # Annotate frame
        for trk in tracks:
            x1, y1, x2, y2, tid = (int(v) for v in trk[:5])
            color = track_color(tid)

            # Try to associate metadata with this track
            # (approximate: closest detection by IoU)
            label = track_class_map.get(tid, "obj")
            conf  = track_conf_map.get(tid, 0.0)

            draw_box(frame, x1, y1, x2, y2, color, f"ID:{tid} {label}", conf)

        # Update metadata maps from current detections (best-effort)
        if len(dets) > 0 and len(tracks) > 0:
            for i, (di, meta) in enumerate(det_meta.items()):
                if di < len(dets):
                    d_box = dets[di, :4]
                    # Find the closest track
                    best_iou, best_tid = -1.0, -1
                    for trk in tracks:
                        t_box = trk[:4].astype(float)
                        iou = _single_iou(d_box, t_box)
                        if iou > best_iou:
                            best_iou, best_tid = iou, int(trk[4])
                    if best_iou > 0.3 and best_tid >= 0:
                        track_class_map[best_tid] = meta[0]
                        track_conf_map[best_tid]  = meta[1]

        # FPS calculation
        ts_history.append(time.perf_counter())
        if len(ts_history) > fps_window:
            ts_history.pop(0)
        fps = (len(ts_history) - 1) / max(ts_history[-1] - ts_history[0], 1e-9) if len(ts_history) > 1 else 0.0

        draw_hud(frame, fps, len(tracks), frame_num, source_label)

        # Display
        cv2.imshow("Object Detection & Tracking", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit requested.")
            break
        elif key == ord("s"):
            fname = f"screenshot_{frame_num:05d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved → {fname}")

    # Cleanup 
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Processed {frame_num} frames.")


#  IoU helper (single pair)

def _single_iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / max(union, 1e-6)

#  CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time Object Detection & Tracking (YOLOv8 + SORT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",       default="0",           help="Video source: '0' = webcam, or path to video file")
    p.add_argument("--model",        default="yolov8n.pt",  help="YOLO model weights (auto-downloaded if absent)")
    p.add_argument("--conf",         type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--nms-iou",      type=float, default=0.45, help="YOLO NMS IoU threshold")
    p.add_argument("--classes",      type=int, nargs="+", default=None, help="Filter COCO class IDs (e.g. 0 2 5)")
    p.add_argument("--save",         default=None,          help="Save annotated video to this path (.mp4)")
    # SORT parameters
    p.add_argument("--max-age",      type=int, default=3,   help="SORT: max frames without detection")
    p.add_argument("--min-hits",     type=int, default=3,   help="SORT: min detections before confirming a track")
    p.add_argument("--iou-threshold",type=float, default=0.3, help="SORT: IoU threshold for matching")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
