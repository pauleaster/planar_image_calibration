"""Image-quality utilities for the planar reconstruction pipeline."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np

CV2 = cast(Any, cv2)


def compute_sharpness_score(frame: np.ndarray) -> float:
    """Compute a blur/sharpness score using variance of the Laplacian.

    Larger values generally indicate sharper frames.
    """
    if frame.size == 0:
        raise ValueError("Frame must be a non-empty numpy array.")

    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3:
        gray = CV2.cvtColor(frame, CV2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Frame must be 2D grayscale or 3D BGR image.")

    return float(CV2.Laplacian(gray, CV2.CV_64F).var())
