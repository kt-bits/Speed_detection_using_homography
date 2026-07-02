"""
homography_speed_estimation.py
Upgrade over the two-tripwire method: maps the entire road plane to real-world
ground coordinates via a homography (bird's-eye) transform, so speed can be
measured continuously and accurately anywhere in frame -- not just between
two fixed lines. This is closer to how production traffic-speed systems work.

SETUP REQUIRED (one-time per camera position):
1. Pick 4 points on the road, visible in your camera frame, that form a
   real-world RECTANGLE -- e.g. the two edges of a lane over some length.
   Example: lane width 3.5m (Indian standard), length 15m along the road.
2. Physically measure or reliably estimate the real width/length of that
   rectangle. Accuracy here directly determines speed accuracy.
3. Run calibrate_homography() once on a sample frame, click the 4 points
   IN ORDER: top-left, top-right, bottom-right, bottom-left (as they appear
   in the image, going around the rectangle). Save the resulting matrix.
4. Reuse the saved matrix in main() for that camera position -- recalibrate
   any time the camera moves or its angle changes.
"""

import cv2
import numpy as np
from collections import deque

# ---- CONFIG ----
VIDEO_SOURCE = "viol_test(1).mp4"
MODEL_PATH = "yolov8n.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
HOMOGRAPHY_FILE = "homography_matrix.npy"

# real-world rectangle dimensions in meters (must match the 4 points you click)
REAL_WIDTH_M = 3.5    # e.g. lane width
REAL_LENGTH_M = 15.0  # e.g. length of road segment you calibrated

STALE_TRACK_FRAMES = 50
SPEED_WINDOW = 5       # smooth speed over this many recent position samples
MIN_WINDOW_FOR_SPEED = 3  # need at least this many samples before trusting a speed


def calibrate_homography(sample_frame, save_path=HOMOGRAPHY_FILE):
    """
    Click 4 points on sample_frame, in order: top-left, top-right,
    bottom-right, bottom-left of a real-world rectangle on the road.
    Computes and saves the homography matrix mapping image pixels -> meters.
    """
    points = []
    display = sample_frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click 4 points: TL, TR, BR, BL (in order)", display)
            print(f"Point {len(points)}: ({x}, {y})")

    cv2.namedWindow("Click 4 points: TL, TR, BR, BL (in order)")
    cv2.setMouseCallback("Click 4 points: TL, TR, BR, BL (in order)", on_click)
    cv2.imshow("Click 4 points: TL, TR, BR, BL (in order)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError(f"Need exactly 4 points, got {len(points)}")

    src_pts = np.array(points, dtype=np.float32)
    # destination: a flat rectangle in real-world meters, origin at top-left
    dst_pts = np.array([
        [0, 0],
        [REAL_WIDTH_M, 0],
        [REAL_WIDTH_M, REAL_LENGTH_M],
        [0, REAL_LENGTH_M],
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    np.save(save_path, H)
    print(f"Homography matrix saved to {save_path}")
    return H


def pixel_to_world(H, px, py):
    """Transform an image pixel (px, py) to real-world (X, Y) in meters."""
    point = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(point, H)
    return world[0][0][0], world[0][0][1]


def main():
    try:
        H = np.load(HOMOGRAPHY_FILE)
    except FileNotFoundError:
        raise RuntimeError(
            f"{HOMOGRAPHY_FILE} not found. Run calibrate_homography() on a "
            "sample frame from this camera first."
        )

    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)

    # per-track-id state: deque of (timestamp, world_x, world_y)
    position_history = {}
    last_seen = {}
    speeds = {}

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    results_gen = model.track(
        source=VIDEO_SOURCE,
        classes=VEHICLE_CLASSES,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for result in results_gen:
        frame = result.orig_img
        now = frame_idx / fps

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                px, py = (x1 + x2) / 2, y2  # bottom-center, on the road plane

                last_seen[track_id] = frame_idx

                wx, wy = pixel_to_world(H, px, py)

                if track_id not in position_history:
                    position_history[track_id] = deque(maxlen=SPEED_WINDOW)
                position_history[track_id].append((now, wx, wy))

                hist = position_history[track_id]
                if len(hist) >= MIN_WINDOW_FOR_SPEED:
                    t_old, x_old, y_old = hist[0]
                    t_new, x_new, y_new = hist[-1]
                    dt = t_new - t_old
                    if dt > 0:
                        dist_m = ((x_new - x_old) ** 2 + (y_new - y_old) ** 2) ** 0.5
                        speed_kmph = round((dist_m / dt) * 3.6, 1)
                        speeds[track_id] = speed_kmph

                label = f"ID {track_id}"
                if track_id in speeds:
                    label += f" | {speeds[track_id]} km/h"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 255), -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # cleanup stale tracks
        stale_ids = [tid for tid, seen in last_seen.items()
                     if frame_idx - seen > STALE_TRACK_FRAMES]
        for tid in stale_ids:
            position_history.pop(tid, None)
            speeds.pop(tid, None)
            last_seen.pop(tid, None)

        cv2.imshow("Homography Speed Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return speeds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true",
                         help="Run one-time calibration first (click 4 points on a sample frame)")
    args = parser.parse_args()

    if args.calibrate:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read a frame from {VIDEO_SOURCE} for calibration.")
        calibrate_homography(frame)
        print("Calibration done. Now run again without --calibrate to start speed estimation.")
    else:
        main()
