# Real-Time Object Detection & Tracking

YOLOv8 detection pipeline paired with a hand-rolled SORT tracker —
no heavy DeepSORT re-ID model needed for most use-cases.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on webcam (model auto-downloads ~6 MB on first run)
python main.py

# 3. Run on a video file and save the output
python main.py --source input.mp4 --save output.mp4

# 4. Use a larger (more accurate) model
python main.py --model yolov8m.pt

# 5. Track only people and cars
python main.py --classes 0 2
```

---

## Key Controls (while window is open)

| Key | Action            |
|-----|-------------------|
| `Q` | Quit              |
| `S` | Save screenshot   |

---

## Project Structure

```
├── main.py           ← Entry point: detection + drawing + video I/O
├── sort_tracker.py   ← Full SORT implementation (Kalman + Hungarian)
├── requirements.txt
└── README.md
```

---

## CLI Arguments

| Argument          | Default        | Description                                    |
|-------------------|----------------|------------------------------------------------|
| `--source`        | `0`            | `0` = webcam, or path to video file            |
| `--model`         | `yolov8n.pt`   | Model weights (auto-downloads from ultralytics)|
| `--conf`          | `0.35`         | YOLO confidence threshold (0–1)                |
| `--nms-iou`       | `0.45`         | YOLO NMS IoU threshold                         |
| `--classes`       | *(all)*        | COCO class IDs to keep (e.g. `0 2 5`)          |
| `--save`          | *(none)*       | Output video path                              |
| `--max-age`       | `3`            | SORT: frames to keep track without detection   |
| `--min-hits`      | `3`            | SORT: hits before a track is confirmed         |
| `--iou-threshold` | `0.3`          | SORT: matching IoU threshold                   |

---

## Available YOLOv8 Models

| Model          | Size   | Speed  | Accuracy |
|----------------|--------|--------|----------|
| `yolov8n.pt`   | 6 MB   | ████░  | ██░░░    |
| `yolov8s.pt`   | 22 MB  | ███░░  | ███░░    |
| `yolov8m.pt`   | 50 MB  | ██░░░  | ████░    |
| `yolov8l.pt`   | 83 MB  | █░░░░  | █████    |
| `yolov8x.pt`   | 130 MB | ░░░░░  | █████    |

---

## Algorithm Overview

```
Frame N
  │
  ▼
┌──────────────┐     ┌──────────────────────────────────┐
│  YOLOv8      │────▶│  Detections [x1,y1,x2,y2, conf]  │
│  (detection) │     └──────────────┬───────────────────┘
└──────────────┘                    │
                                    ▼
                       ┌────────────────────────┐
                       │  SORT Tracker          │
                       │  ① Kalman predict      │
                       │  ② IoU cost matrix     │
                       │  ③ Hungarian assign    │
                       │  ④ Kalman update       │
                       │  ⑤ Birth / death logic │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       Tracks [x1,y1,x2,y2, ID]
                                   │
                                   ▼
                          Draw & Display
```

---

## Common COCO Class IDs

| ID | Class     | ID | Class       |
|----|-----------|----|-------------|
| 0  | person    | 14 | bird        |
| 1  | bicycle   | 15 | cat         |
| 2  | car       | 16 | dog         |
| 3  | motorcycle| 24 | backpack    |
| 5  | bus       | 26 | handbag     |
| 7  | truck     | 41 | cup         |
| 9  | traffic light | 62 | tv       |
| 11 | stop sign | 67 | cell phone  |

---

## Tuning Tips

- **Many false tracks?** → increase `--min-hits` (3 → 5) and `--conf` (0.35 → 0.5)
- **Tracks lost too quickly?** → increase `--max-age` (3 → 10)
- **ID switches in dense scenes?** → decrease `--iou-threshold` (0.3 → 0.2)
- **Slow on CPU?** → use `yolov8n.pt`; halve resolution with `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`
