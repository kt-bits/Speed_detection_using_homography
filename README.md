# Homography Speed Estimation
 
Estimate vehicle speeds from a single fixed camera by mapping the entire road
plane to real-world ground coordinates with a homography (bird's-eye)
transform. This is an upgrade over a simple two-tripwire method: instead of
only measuring speed between two fixed lines, every tracked vehicle gets a
continuous real-world position, so speed can be computed anywhere in frame.
 
## How it works
 
1. **Detection & tracking** — [Ultralytics YOLOv8](https://docs.ultralytics.com/)
   detects vehicles (car, motorcycle, bus, truck) and [ByteTrack](https://github.com/ifzhang/ByteTrack)
   assigns each one a persistent track ID across frames.
2. **Homography mapping** — A one-time calibration maps a real-world
   rectangle on the road (e.g. a lane segment) from image pixels to metric
   ground coordinates (meters). Every subsequent detection's pixel position
   is transformed into this metric space.
3. **Speed calculation** — For each track, a rolling window of recent
   `(timestamp, world_x, world_y)` samples is kept. Speed is computed from
   the straight-line distance between the oldest and newest sample in the
   window, divided by the elapsed time, converted to km/h.
## Requirements
 
```bash
pip install ultralytics opencv-python numpy
```
 
You'll also need:
- A video file of the road segment (`input_video.mp4` by default)
- A YOLOv8 weights file (`yolov8n.pt` by default — downloads automatically
  via Ultralytics if not present)
## One-time setup: calibration
 
Speed accuracy depends entirely on this step, so take care with your
measurements.
 
1. Pick 4 points on the road, visible in your camera frame, that form a
   real-world **rectangle** — for example, the two edges of a lane over some
   length (a lane width of 3.5 m and a length of 15 m are used as defaults,
   matching common Indian lane widths).
2. Physically measure (or reliably estimate) the real width and length of
   that rectangle in meters. Update `REAL_WIDTH_M` and `REAL_LENGTH_M` in the
   config section of the script to match.
3. Run calibration on a sample frame from your video:
```bash
   python homography_speed_estimation.py --calibrate
```
 
   A window will open showing the first frame of `VIDEO_SOURCE`. Click the 4
   points **in order**: top-left, top-right, bottom-right, bottom-left, as
   they appear in the image, going around the rectangle. Press any key once
   done.
4. The resulting homography matrix is saved to `homography_matrix.npy` and
   reused automatically on future runs.
 
**Recalibrate any time the camera moves or its angle/zoom changes.**
 
## Running speed estimation
 
Once calibrated:
 
```bash
python homography_speed_estimation.py
```
 
This opens a display window showing each tracked vehicle with its bounding
box, ground-contact point (the small yellow dot, tracked at the bottom-center
of the box), track ID, and current speed in km/h once enough samples have
been collected. Press `q` to quit early.
 
The script returns a dictionary of `{track_id: speed_kmph}` for the vehicles
tracked when it exits.
 
## Configuration
 
All key parameters live at the top of the script:
 
| Variable | Description | Default |
|---|---|---|
| `VIDEO_SOURCE` | Path to input video | `input_video.mp4` |
| `MODEL_PATH` | YOLOv8 weights file | `yolov8n.pt` |
| `VEHICLE_CLASSES` | COCO class IDs to track (car, motorcycle, bus, truck) | `[2, 3, 5, 7]` |
| `HOMOGRAPHY_FILE` | Saved calibration matrix path | `homography_matrix.npy` |
| `REAL_WIDTH_M` | Real-world width of the calibration rectangle (meters) | `3.5` |
| `REAL_LENGTH_M` | Real-world length of the calibration rectangle (meters) | `15.0` |
| `STALE_TRACK_FRAMES` | Frames of no detection before a track is dropped | `50` |
| `SPEED_WINDOW` | Number of recent position samples used to smooth speed | `5` |
| `MIN_WINDOW_FOR_SPEED` | Minimum samples required before trusting a speed value | `3` |
 
## Accuracy notes
 
- Speed accuracy is only as good as your calibration measurements — errors in
  the real-world width/length propagate directly into speed error.
- The calibration rectangle should be as large as practical within the frame
  (covering more of the road vehicles actually drive through) to reduce the
  effect of pixel-level detection noise on the homography mapping.
- The bottom-center of each bounding box is used as the vehicle's
  ground-contact point; this assumes a relatively flat road and a camera
  angle where that point closely approximates where the vehicle touches the
  road.
- Speed is smoothed over `SPEED_WINDOW` samples to reduce jitter from
  frame-to-frame detection noise; increase this for smoother but slower-to-
  update readings, decrease it for more responsive but noisier readings.
## Limitations
 
- Assumes a flat road plane — significant slopes or elevation changes will
  distort the mapping.
- Camera must remain fixed after calibration.
- Detection/tracking quality (occlusion, poor lighting, fast motion) directly
  affects speed accuracy, since it relies on YOLOv8 + ByteTrack for per-frame
  positions.# Speed_detection_using_homography
