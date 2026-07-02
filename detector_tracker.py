"""
detector_tracker.py
--------------------
Core object detection and tracking logic for the CodeAlpha Object Detection
and Tracking project (Task 4).

This module wraps the Ultralytics YOLOv8 model together with its built-in
multi-object tracker (ByteTrack / BoT-SORT). Heavy third-party imports
(cv2, numpy, ultralytics) are performed *inside* functions so that this
module can be imported / py_compiled cleanly on machines where those heavy
packages are not installed.

Author: CodeAlpha AI Internship - Task 4
"""

import os


# Video containers we are willing to accept from the user.
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}


def is_allowed(filename):
    """
    Return True if `filename` has an extension we support.

    Parameters
    ----------
    filename : str
        The file name (or full path) to validate.

    Returns
    -------
    bool
        True when the extension is one of ALLOWED_EXTENSIONS, else False.
    """
    if not filename or not isinstance(filename, str):
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


class DetectorTracker:
    """
    Detect and track objects in a video using YOLOv8 + its built-in tracker.

    The YOLO model is lazy-loaded on first use so that constructing this
    object is cheap and does not require the heavy `ultralytics` dependency
    until detection actually runs.

    Parameters
    ----------
    model_path : str
        Path (or model name) passed to Ultralytics YOLO. The default
        'yolov8n.pt' auto-downloads from Ultralytics on first run and
        therefore requires an internet connection the first time.
    conf : float
        Confidence threshold in the range [0.0, 1.0]. Detections below this
        confidence are ignored.
    """

    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        # Basic validation of the confidence threshold.
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            raise ValueError("`conf` must be a number between 0.0 and 1.0.")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("`conf` must be between 0.0 and 1.0.")

        self.model_path = model_path
        self.conf = conf
        # The actual YOLO model is loaded lazily in `_load_model`.
        self._model = None

    def _load_model(self):
        """
        Lazily import Ultralytics and load the YOLO model.

        Raises
        ------
        RuntimeError
            If Ultralytics is not installed or the model fails to load /
            download. The message explains exactly what is required.
        """
        if self._model is not None:
            return self._model

        # Import inside the function so the module stays importable without
        # the heavy dependency present.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The 'ultralytics' package is not installed. Install project "
                "dependencies first with:\n\n    pip install -r requirements.txt\n"
            ) from exc

        try:
            self._model = YOLO(self.model_path)
        except Exception as exc:  # noqa: BLE001 - surface a friendly message
            raise RuntimeError(
                f"Failed to load the YOLO model '{self.model_path}'.\n"
                "On the first run the weights (e.g. yolov8n.pt) auto-download "
                "from Ultralytics, which requires an active internet "
                "connection. If you are offline, download the .pt file "
                "manually and pass its local path as `model_path`.\n"
                f"Original error: {exc}"
            ) from exc

        return self._model

    def process_video(self, input_path, output_path, progress_callback=None):
        """
        Run detection + tracking over every frame of `input_path` and write
        an annotated video to `output_path`.

        Parameters
        ----------
        input_path : str
            Path to the source video. Must exist and be readable by OpenCV.
        output_path : str
            Path where the annotated MP4 will be written. Parent directories
            are created if needed.
        progress_callback : callable, optional
            Called as `progress_callback(current_frame, total_frames)` after
            each frame so a UI can update a progress bar.

        Returns
        -------
        tuple(str, dict)
            The `output_path` and a small stats dict:
            {"frames": <int>, "unique_ids": <int>}.

        Raises
        ------
        FileNotFoundError
            If `input_path` does not exist.
        ValueError
            If `input_path` has an unsupported extension.
        RuntimeError
            If the video cannot be opened, the writer cannot be created,
            or the model / inference fails.
        """
        # Import heavy libs inside the method (keeps module import light).
        import cv2

        # ---- Input validation -------------------------------------------------
        if not input_path or not os.path.isfile(input_path):
            raise FileNotFoundError(
                f"Input video not found: '{input_path}'. Please provide a "
                "valid video file path."
            )
        if not is_allowed(input_path):
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type for '{input_path}'. "
                f"Allowed extensions: {allowed}."
            )

        # Ensure the model is ready (may raise a clear RuntimeError).
        model = self._load_model()

        # Make sure the output directory exists.
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        # ---- Open the source video -------------------------------------------
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open the video '{input_path}'. The file may "
                "be corrupt, or your OpenCV build may lack the codec required "
                "to decode it."
            )

        writer = None
        unique_ids = set()
        frame_count = 0

        try:
            # Read video properties. Fall back to sane defaults if missing.
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0  # reasonable default when FPS metadata is absent
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 0  # unknown; progress will just report current

            # ---- Set up the output writer (mp4v codec) -----------------------
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not create the output video writer for "
                    f"'{output_path}'. Check that the folder is writable and "
                    "that your OpenCV build supports the mp4v codec."
                )

            # ---- Process frame by frame --------------------------------------
            while True:
                ok, frame = cap.read()
                if not ok:
                    break  # end of stream

                # Run detection + tracking. `persist=True` keeps track IDs
                # consistent across frames. Errors here are surfaced clearly.
                try:
                    results = model.track(
                        frame,
                        persist=True,
                        conf=self.conf,
                        verbose=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "Object tracking/inference failed while processing the "
                        f"video (frame {frame_count + 1}). Original error: {exc}"
                    ) from exc

                # Annotate the frame with our own drawing so we control labels.
                annotated = self._annotate_frame(cv2, frame, results, unique_ids)

                writer.write(annotated)
                frame_count += 1

                # Report progress to the caller (e.g. Streamlit progress bar).
                if progress_callback is not None:
                    try:
                        progress_callback(frame_count, total_frames)
                    except Exception:
                        # Never let a UI callback crash the pipeline.
                        pass

            if frame_count == 0:
                raise RuntimeError(
                    "No frames could be read from the video. The file may be "
                    "empty or in an unsupported format."
                )

        finally:
            # Always release resources, even if an error occurred.
            cap.release()
            if writer is not None:
                writer.release()

        stats = {"frames": frame_count, "unique_ids": len(unique_ids)}
        return output_path, stats

    def _annotate_frame(self, cv2, frame, results, unique_ids):
        """
        Draw bounding boxes, class names, confidences and tracking IDs onto
        a copy-free reference of `frame` (OpenCV draws in place).

        Parameters
        ----------
        cv2 : module
            The already-imported OpenCV module (passed to avoid re-importing).
        frame : numpy.ndarray
            The current BGR frame.
        results : list
            Output of `model.track(...)` for this single frame.
        unique_ids : set
            Accumulator of all tracking IDs seen so far (mutated in place).

        Returns
        -------
        numpy.ndarray
            The annotated frame.
        """
        # `results` is a list with a single Results object for one image.
        if not results:
            return frame

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return frame

        # Class-index -> human-readable name mapping from the model.
        names = getattr(result, "names", {}) or {}

        for box in boxes:
            # Coordinates as ints. `.xyxy` is a tensor of shape (1, 4).
            try:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            except Exception:
                continue  # skip malformed boxes defensively

            # Confidence score.
            conf = float(box.conf[0]) if box.conf is not None else 0.0

            # Class id -> name.
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            label_name = names.get(cls_id, str(cls_id))

            # Tracking id (may be None on the very first frame of a track).
            track_id = None
            if getattr(box, "id", None) is not None:
                try:
                    track_id = int(box.id[0])
                    unique_ids.add(track_id)
                except Exception:
                    track_id = None

            id_text = f"ID {track_id}" if track_id is not None else "ID -"
            label = f"{id_text} | {label_name} {conf:.2f}"

            # Draw the bounding box.
            color = (0, 200, 0)  # green in BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw a filled label background for readability.
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            top = max(0, y1 - th - baseline - 4)
            cv2.rectangle(
                frame,
                (x1, top),
                (x1 + tw + 4, y1),
                color,
                thickness=-1,  # filled
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # black text on green background
                1,
                cv2.LINE_AA,
            )

        return frame
