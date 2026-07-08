"""Feature detection and matching utilities for planar reconstruction.

This module provides a small OpenCV-backed ORB feature detector and matcher so
feature logic stays separate from the CLI, stream acquisition, and later UI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureSet:
    """Detected keypoints and descriptors for a single image."""

    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray | None


@dataclass(frozen=True)
class MatchResult:
    """Matched feature pairs between two images."""

    matches: list[cv2.DMatch]
    source_points: np.ndarray
    target_points: np.ndarray


def create_orb_detector(max_features: int = 1000) -> cv2.ORB:
    """Create a configured ORB detector."""
    if max_features <= 0:
        raise ValueError("max_features must be greater than zero.")
    detector = cast(cv2.ORB, cv2.ORB_create(nfeatures=max_features)) # type: ignore
    return detector


def detect_features(image: np.ndarray, *, max_features: int = 1000) -> FeatureSet:
    """Detect ORB features in an image."""
    if image.size == 0:
        raise ValueError("image must be a non-empty numpy array.")

    detector = create_orb_detector(max_features=max_features)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return FeatureSet(keypoints=list(keypoints), descriptors=descriptors)


def match_features(
    source: FeatureSet,
    target: FeatureSet,
    *,
    ratio_threshold: float = 0.75,
) -> MatchResult:
    """Match features from a source image to a target image using a ratio test."""
    if ratio_threshold <= 0.0 or ratio_threshold >= 1.0:
        raise ValueError("ratio_threshold must be between 0 and 1.")
    if source.descriptors is None or target.descriptors is None:
        return MatchResult(
            matches=[],
            source_points=np.empty((0, 2), dtype=np.float32),
            target_points=np.empty((0, 2), dtype=np.float32),
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(source.descriptors, target.descriptors, k=2)

    good_matches: list[cv2.DMatch] = []
    source_points: list[tuple[float, float]] = []
    target_points: list[tuple[float, float]] = []

    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second_best = pair
        if best.distance < ratio_threshold * second_best.distance:
            good_matches.append(best)
            sx, sy = source.keypoints[best.queryIdx].pt
            tx, ty = target.keypoints[best.trainIdx].pt
            source_points.append((float(sx), float(sy)))
            target_points.append((float(tx), float(ty)))

    return MatchResult(
        matches=good_matches,
        source_points=np.asarray(source_points, dtype=np.float32),
        target_points=np.asarray(target_points, dtype=np.float32),
    )
