"""
app.py
------
Streamlit demo UI for the CodeAlpha Object Detection and Tracking project
(Task 4).

The user uploads a short video, picks a confidence threshold, and the app
runs YOLOv8 + built-in tracking, then plays back and offers to download the
annotated result.

Run with:
    streamlit run app.py

Author: CodeAlpha AI Internship - Task 4
"""

import os
import tempfile

import streamlit as st

# Import our core logic. This module keeps heavy libraries (cv2, ultralytics)
# inside functions, so importing it here is cheap and safe.
from detector_tracker import DetectorTracker, is_allowed, ALLOWED_EXTENSIONS

# Folder where annotated outputs are written.
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _safe_name(filename):
    """Return just the base name of an uploaded file (strip any path parts)."""
    return os.path.basename(filename).replace(" ", "_")


def main():
    st.set_page_config(
        page_title="Object Detection & Tracking",
        page_icon="🎯",
        layout="centered",
    )

    st.title("🎯 Object Detection & Tracking")
    st.write(
        "Upload a short video and this app will detect objects with **YOLOv8** "
        "and track each one across frames, drawing bounding boxes, class names, "
        "confidence scores and a stable **tracking ID** for every object."
    )
    st.caption(
        "Built for the CodeAlpha AI Internship - Task 4. "
        "On the first run the YOLOv8 weights (`yolov8n.pt`) auto-download from "
        "Ultralytics, which needs an internet connection."
    )

    st.divider()

    # ---- File upload ------------------------------------------------------
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        help="Accepted formats: mp4, avi, mov.",
    )

    # ---- Confidence slider ------------------------------------------------
    conf = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.30,
        step=0.05,
        help="Detections below this confidence are ignored. Higher = stricter.",
    )

    # ---- Run button -------------------------------------------------------
    run = st.button("Run detection & tracking", type="primary")

    if not run:
        st.info("Upload a video and click **Run detection & tracking** to start.")
        _webcam_note()
        return

    # ---- Validate input on run -------------------------------------------
    if uploaded is None:
        st.error("Please upload a video file first.")
        return

    if not is_allowed(uploaded.name):
        allowed = ", ".join(sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS))
        st.error(f"Unsupported file type. Please upload one of: {allowed}.")
        return

    # Persist the upload to a temp file so OpenCV can read it from disk.
    suffix = os.path.splitext(uploaded.name)[1].lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            input_path = tmp.name
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not save the uploaded file to disk: {exc}")
        return

    # Prepare the output path.
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    base = _safe_name(uploaded.name)
    stem = os.path.splitext(base)[0]
    output_path = os.path.join(OUTPUTS_DIR, f"annotated_{stem}.mp4")

    # ---- Progress bar wiring ---------------------------------------------
    progress_bar = st.progress(0, text="Starting...")

    def progress_callback(current, total):
        if total and total > 0:
            frac = min(max(current / total, 0.0), 1.0)
            progress_bar.progress(
                frac, text=f"Processing frame {current} / {total}"
            )
        else:
            # Unknown total: show an indeterminate-style message.
            progress_bar.progress(0, text=f"Processing frame {current}...")

    # ---- Run the pipeline -------------------------------------------------
    try:
        with st.spinner("Loading model and processing video. This may take a while..."):
            tracker = DetectorTracker(conf=conf)
            result_path, stats = tracker.process_video(
                input_path, output_path, progress_callback=progress_callback
            )
    except FileNotFoundError as exc:
        st.error(str(exc))
        _cleanup(input_path)
        return
    except ValueError as exc:
        st.error(str(exc))
        _cleanup(input_path)
        return
    except RuntimeError as exc:
        # These carry the friendly, actionable messages (model download,
        # codec, inference failures, etc.).
        st.error(str(exc))
        _cleanup(input_path)
        return
    except Exception as exc:  # noqa: BLE001 - last-resort friendly message
        st.error(
            "An unexpected error occurred while processing the video:\n\n"
            f"{exc}"
        )
        _cleanup(input_path)
        return

    progress_bar.progress(1.0, text="Done!")

    # ---- Success output ---------------------------------------------------
    st.success(
        f"Processing complete. Analysed {stats['frames']} frames and tracked "
        f"{stats['unique_ids']} unique object(s)."
    )

    if os.path.isfile(result_path):
        # Show the annotated video.
        try:
            with open(result_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(
                label="Download annotated video",
                data=video_bytes,
                file_name=os.path.basename(result_path),
                mime="video/mp4",
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(
                "The video was processed and saved to "
                f"`{result_path}`, but it could not be displayed here: {exc}"
            )
        st.caption(f"Saved to: `{result_path}`")
    else:
        st.error(
            "Processing reported success but the output file was not found at "
            f"`{result_path}`."
        )

    _cleanup(input_path)
    _webcam_note()


def _webcam_note():
    """Show a note about webcam limitations in Streamlit."""
    st.divider()
    st.caption(
        "Note: live webcam streaming is limited in Streamlit and is not the "
        "supported path here. Please use the video-upload workflow above."
    )


def _cleanup(path):
    """Best-effort removal of the temporary uploaded file."""
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        # Non-fatal: temp files are cleaned up by the OS eventually.
        pass


if __name__ == "__main__":
    main()
