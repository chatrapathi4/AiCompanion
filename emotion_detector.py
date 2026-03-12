"""
emotion_detector.py
--------------------
Real-time facial emotion detection using OpenCV and FER.

Captures frames from the webcam, detects faces, and classifies
the dominant emotion (happy, sad, angry, surprise, fear, disgust, neutral).
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from logger_setup import get_logger

try:
    from fer import FER
except (ImportError, ModuleNotFoundError):
    from fer.fer import FER

log = get_logger(__name__)


class EmotionDetector:
    """Wraps OpenCV video capture and FER emotion analysis."""

    def __init__(self, camera_index: int = 0):
        """
        Initialise the emotion detector.

        Args:
            camera_index: Index of the webcam device (default 0).
        """
        self._camera_index = camera_index
        self._cap = None
        # Use MTCNN detector for better accuracy; falls back to OpenCV Haar cascade
        self._detector = FER(mtcnn=True)
        # Attempt to help OpenCV window responsiveness on some systems
        try:
            cv2.startWindowThread()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def start_camera(self) -> None:
        """Open the webcam. Raises RuntimeError if the camera is unavailable."""
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self._camera_index}. "
                "Please check that a webcam is connected and not in use by another application."
            )
        log.info("Camera opened (index %d).", self._camera_index)

    def release(self) -> None:
        """Release the webcam and close any OpenCV windows."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            cv2.destroyAllWindows()
            log.info("Camera released.")

    # ------------------------------------------------------------------
    # Emotion detection
    # ------------------------------------------------------------------

    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the webcam.

        Returns:
            The captured frame (BGR numpy array).

        Raises:
            RuntimeError: If the camera is not started or the frame grab fails.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not started. Call start_camera() first.")

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from the camera.")
        return frame

    def detect_emotion(self, frame: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """Detect the dominant emotion in a frame.

        Args:
            frame: The frame to analyze. If None, captures a new one.

        Returns:
            A tuple of (emotion_label, confidence).
        """
        if frame is None:
            try:
                frame = self.capture_frame()
            except RuntimeError as exc:
                log.warning("Capture error: %s", exc)
                return ("neutral", 0.0)

        try:
            result = self._detector.top_emotion(frame)
        except Exception as exc:
            log.warning("FER analysis error: %s", exc)
            return ("neutral", 0.0)

        if result is None or result[0] is None:
            # No face detected in the frame
            return ("neutral", 0.0)

        emotion, confidence = result
        # Note: We keep the logging minimal as this may be called frequently
        return (emotion, float(confidence))

    def show_preview(self, frame: np.ndarray, label: str = "", confidence: float = 0.0, duration_ms: int = 1) -> None:
        """Display the camera feed in a window with an optional emotion label.

        Args:
            frame:       The frame to display (BGR numpy array).
            label:       Text label (e.g., emotion name).
            confidence:  Confidence score (0.0 to 1.0).
            duration_ms: waitKey delay in milliseconds (default 1).
        """
        if frame is None:
            return

        # Draw the label on the frame for visual feedback
        if label:
            text = f"{label.capitalize()} ({confidence:.2f})"
            cv2.putText(
                frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        cv2.imshow("Emotion Detector Preview", frame)
        cv2.waitKey(duration_ms)
