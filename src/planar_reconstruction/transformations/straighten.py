"""Perspective-straightening helpers decoupled from the UI layer."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class StraightenTransformResult:
    """Result of rectifying a frame while keeping full-frame warped content."""

    warped_image: npt.NDArray[np.uint8]
    homography: npt.NDArray[np.float64]
    ordered_corners: npt.NDArray[np.float32]


def order_corners(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    if points.shape != (4, 2):
        raise ValueError("Exactly four 2D points are required.")

    sums = points.sum(axis=1)
    diffs = points[:, 0] - points[:, 1]

    ordered: npt.NDArray[np.float32] = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right
    ordered[1] = points[np.argmax(diffs)]  # top-right
    ordered[3] = points[np.argmin(diffs)]  # bottom-left
    return ordered


def straighten_full_frame(
    image: npt.NDArray[np.uint8],
    selected_points: npt.NDArray[np.float32],
    preserve_corners: npt.NDArray[np.float32] | None = None,
) -> StraightenTransformResult:
    """Rectify using selected corners and warp the full frame onto an expanded canvas.

    The selected quadrilateral is mapped to a rectangle, but unlike a simple crop
    rectification, the warp is applied to the entire source frame and the output
    canvas is expanded so transformed pixels outside the selected corners are kept.
    """
    if selected_points.shape != (4, 2):
        raise ValueError("Exactly four selected corner points are required.")

    ordered = order_corners(selected_points.astype(np.float32))

    dx_top = float(ordered[1, 0] - ordered[0, 0])
    dy_top = float(ordered[1, 1] - ordered[0, 1])
    dx_bottom = float(ordered[2, 0] - ordered[3, 0])
    dy_bottom = float(ordered[2, 1] - ordered[3, 1])
    dx_left = float(ordered[3, 0] - ordered[0, 0])
    dy_left = float(ordered[3, 1] - ordered[0, 1])
    dx_right = float(ordered[2, 0] - ordered[1, 0])
    dy_right = float(ordered[2, 1] - ordered[1, 1])

    width_top = (dx_top * dx_top + dy_top * dy_top) ** 0.5
    width_bottom = (dx_bottom * dx_bottom + dy_bottom * dy_bottom) ** 0.5
    height_left = (dx_left * dx_left + dy_left * dy_left) ** 0.5
    height_right = (dx_right * dx_right + dy_right * dy_right) ** 0.5

    rect_width = int(round(max(width_top, width_bottom)))
    rect_height = int(round(max(height_left, height_right)))
    if rect_width < 2 or rect_height < 2:
        raise ValueError("Selected corners produce an invalid output size.")

    destination: npt.NDArray[np.float32] = np.array(
        [
            [0.0, 0.0],
            [rect_width - 1.0, 0.0],
            [rect_width - 1.0, rect_height - 1.0],
            [0.0, rect_height - 1.0],
        ],
        dtype=np.float32,
    )

    base_h = cv2.getPerspectiveTransform(ordered, destination)

    if preserve_corners is None:
        source_h, source_w = image.shape[:2]
        full_corners: npt.NDArray[np.float32] = np.array(
            [
                [0.0, 0.0],
                [source_w - 1.0, 0.0],
                [source_w - 1.0, source_h - 1.0],
                [0.0, source_h - 1.0],
            ],
            dtype=np.float32,
        )
    else:
        if preserve_corners.shape != (4, 2):
            raise ValueError("Preserve corners must be shape (4, 2).")
        full_corners = preserve_corners.astype(np.float32)

    transformed_full = cv2.perspectiveTransform(
        full_corners.reshape((1, 4, 2)), base_h
    ).reshape(4, 2)

    min_x = float(np.min(transformed_full[:, 0]))
    min_y = float(np.min(transformed_full[:, 1]))
    max_x = float(np.max(transformed_full[:, 0]))
    max_y = float(np.max(transformed_full[:, 1]))

    translate_x = -min_x if min_x < 0.0 else 0.0
    translate_y = -min_y if min_y < 0.0 else 0.0

    translate_h: npt.NDArray[np.float64] = np.array(
        [[1.0, 0.0, translate_x], [0.0, 1.0, translate_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    full_h: npt.NDArray[np.float64] = translate_h @ base_h

    out_width = int(np.ceil(max_x + translate_x)) + 1
    out_height = int(np.ceil(max_y + translate_y)) + 1
    out_width = max(2, out_width)
    out_height = max(2, out_height)

    warped = cv2.warpPerspective(image, full_h, (out_width, out_height))
    warped_uint8 = np.ascontiguousarray(warped, dtype=np.uint8)

    return StraightenTransformResult(
        warped_image=warped_uint8,
        homography=full_h,
        ordered_corners=ordered,
    )
