"""Homography estimation utilities for planar reconstruction.

This module stays independent of the CLI and UI so homography logic can be
reused by the reconstruction pipeline and validated in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HomographyResult:
    """Estimated planar transform and basic quality metrics."""

    matrix: np.ndarray | None
    inlier_mask: np.ndarray | None
    inlier_count: int
    inlier_ratio: float


def estimate_homography(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    ransac_reproj_threshold: float = 5.0,
) -> HomographyResult:
    """Estimate a homography from matched point pairs using RANSAC.

    Args:
        source_points: Matched source points with shape (N, 2).
        target_points: Matched target points with shape (N, 2).
        ransac_reproj_threshold: Maximum allowed reprojection error for RANSAC.
    """
    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError("source_points must have shape (N, 2).")
    if target_points.ndim != 2 or target_points.shape[1] != 2:
        raise ValueError("target_points must have shape (N, 2).")
    if source_points.shape[0] != target_points.shape[0]:
        raise ValueError("source_points and target_points must have the same length.")
    if source_points.shape[0] < 4:
        raise ValueError("At least four point pairs are required.")
    if ransac_reproj_threshold <= 0:
        raise ValueError("ransac_reproj_threshold must be greater than zero.")

    homography_matrix, inlier_mask = cv2.findHomography(  # pylint: disable=no-member
        source_points.astype(np.float32),
        target_points.astype(np.float32),
        cv2.RANSAC,  # pylint: disable=no-member
        ransac_reproj_threshold,
    )

    if inlier_mask is None:
        inlier_count = 0
        inlier_ratio = 0.0
    else:
        inlier_count = int(inlier_mask.astype(bool).sum())
        inlier_ratio = inlier_count / float(source_points.shape[0])

    return HomographyResult(
        matrix=homography_matrix,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
    )
