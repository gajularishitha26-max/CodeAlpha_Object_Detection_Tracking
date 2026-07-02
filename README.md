# CodeAlpha Object Detection & Tracking

## 1. Project Title
**CodeAlpha Object Detection & Tracking** - Detect and track multiple objects
in a video using YOLOv8 and its built-in multi-object tracker.

## 2. Objective
Build a Python application that takes a video, detects objects in every frame
with a deep-learning model (YOLOv8), and assigns a **stable tracking ID** to
each object so it can be followed across frames. The annotated video (bounding
boxes, class names, confidence scores and tracking IDs) is produced as output
and viewable through a simple Streamlit web interface.

## 3. Features
- Upload a video (`mp4`, `avi`, `mov`) through a Streamlit UI.
- Object detection with the YOLOv8 model (`yolov8n.pt`).
- Multi-object tracking with Ultralytics' built-in tracker (`model.track`),
  keeping consistent IDs across frames via `persist=True`.
- Adjustable confidence threshold slider.
- Live progress bar while the video is processed.
- Annotated output video with boxes, class name, confidence, and tracking ID.
- In-browser playback plus a download button for the result.
- Summary stats: number of frames processed and count of unique tracked objects.
- Friendly, actionable error messages (missing model/internet, bad file,
  codec issues) - never a raw crash.

## 4. Tech Stack
- **Python**
- **Streamlit** - demo web UI
- **Ultralytics YOLOv8** - detection + tracking
- **OpenCV (opencv-python)** - video I/O and drawing
- **NumPy** - array handling

## 5. Folder Structure
```
CodeAlpha_Object_Detection_Tracking/
├── app.py                 # Streamlit UI (upload, run, playback, download)
├── detector_tracker.py    # DetectorTracker class + detection/tracking logic
├── requirements.txt       # Project dependencies
├── README.md              # This file
├── sample_videos/         # Place your test videos here
│   └── .gitkeep
└── outputs/               # Annotated videos are written here
    └── .gitkeep
```

## 6. Installation Steps
1. (Recommended) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
> Note: `yolov8n.pt` auto-downloads from Ultralytics on the **first run**, so
> an internet connection is required the first time you run detection. If you
> are offline, download `yolov8n.pt` manually and place it next to `app.py`.

## 7. Run Command
```powershell
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## 8. Input and Output
- **Input:** a video file (`mp4`, `avi`, `mov`) uploaded via the web UI, plus a
  chosen confidence threshold.
- **Output:** an annotated MP4 saved to `outputs/annotated_<name>.mp4`, shown in
  the browser with a download button, along with the number of frames processed
  and the count of unique tracked objects.

## 9. Screenshots
_Add screenshots here after running._

## 10. Known Limitations
- The default `yolov8n.pt` is the smallest/fastest model; accuracy is limited
  compared with larger YOLOv8 variants.
- Processing runs frame-by-frame on the CPU by default and can be slow for long
  or high-resolution videos.
- Live webcam streaming is limited in Streamlit and is intentionally not the
  supported path; use video upload instead.
- Output uses the `mp4v` codec; some browsers may need the downloaded file to
  play it, depending on the OpenCV build.

## 11. Future Improvements
- Allow selecting larger YOLOv8 models (`s`/`m`/`l`/`x`) from the UI.
- Add GPU acceleration and batch inference for speed.
- Let the user filter which object classes to detect/track.
- Export per-object trajectories and counts to CSV.
- Add a proper webcam/RTSP live mode using `streamlit-webrtc`.

## 12. CodeAlpha Submission Note
This project was built for the **CodeAlpha Artificial Intelligence Internship -
Task 4: Object Detection and Tracking**.
