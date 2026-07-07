"""Tests for homography estimation behavior."""

from __future__ import annotations

import numpy as np

from planar_reconstruction.homography import estimate_homography


def test_estimate_homography_identity_case_has_all_inliers() -> None:
    """Matching identical points should estimate a valid near-identity transform."""
    points = np.array(
        [
            [10.0, 20.0],
            [110.0, 20.0],
            [10.0, 120.0],
            [110.0, 120.0],
            [60.0, 70.0],
            [80.0, 90.0],
        ],
        dtype=np.float32,
    )

    result = estimate_homography(points, points)

    assert result.matrix is not None
    assert result.inlier_mask is not None
    assert result.inlier_count == points.shape[0]
    assert result.inlier_ratio == 1.0


def test_estimate_homography_requires_at_least_four_points() -> None:
    """RANSAC homography requires at least four point pairs."""
    source = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    target = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)

    try:
        estimate_homography(source, target)
        assert False, "Expected ValueError for insufficient points."
    except ValueError as exc:
        assert "At least four point pairs" in str(exc)
