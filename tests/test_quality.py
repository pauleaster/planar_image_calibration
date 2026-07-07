"""Tests for image sharpness scoring helpers."""

from __future__ import annotations

import numpy as np

from planar_reconstruction.quality import compute_sharpness_score


def test_compute_sharpness_score_higher_for_sharp_pattern() -> None:
    """A structured checkerboard should score sharper than a flat image."""
    checker = np.indices((64, 64)).sum(axis=0) % 2
    checker = (checker * 255).astype(np.uint8)
    flat = np.full((64, 64), 127, dtype=np.uint8)

    checker_score = compute_sharpness_score(checker)
    flat_score = compute_sharpness_score(flat)

    assert checker_score > flat_score


def test_compute_sharpness_score_rejects_empty_frame() -> None:
    """An empty frame should raise a clear value error."""
    empty = np.array([], dtype=np.uint8)

    try:
        compute_sharpness_score(empty)
        assert False, "Expected ValueError for empty frame."
    except ValueError as exc:
        assert "non-empty" in str(exc)
