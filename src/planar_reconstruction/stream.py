"""Video frame streaming helpers for the planar reconstruction pipeline.

This module keeps acquisition logic separate from the CLI and future UI code.
It will later be reused for deterministic video-file replay and, potentially,
live camera or SDK-backed frame sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """Container for a single video frame and its metadata."""

    index: int
    timestamp_ms: float
    frame: np.ndarray


def iter_video_frames(
    video_path: str | Path,
    *,
    max_frames: int | None = None,
    frame_step: int = 1,
) -> Iterator[FramePacket]:
    """Yield frames from a video file in capture order.

    Args:
        video_path: Path to the video file.
        max_frames: Optional maximum number of frames to read.
        frame_step: Yield every Nth frame. Must be greater than zero.
    """
    if frame_step <= 0:
        raise ValueError("frame_step must be greater than zero.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero when provided.")

    capture = cv2.VideoCapture(str(video_path))  # pylint: disable=no-member
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        frames_read = 0
        while max_frames is None or frames_read < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))  # pylint: disable=no-member
            if frames_read % frame_step == 0:
                yield FramePacket(
                    index=frames_read,
                    timestamp_ms=timestamp_ms,
                    frame=frame,
                )

            frames_read += 1
    finally:
        capture.release()
