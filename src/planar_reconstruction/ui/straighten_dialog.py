"""Interactive dialog for selecting corners and straightening a planar region."""

# pylint: disable=no-member

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from planar_reconstruction.transformations.straighten import straighten_full_frame


class CornerSelectionCanvas(QWidget):
    """Canvas for bounding-box drawing, free-order corner picks, and magnifier."""

    bounding_rect_defined = Signal(object)
    corner_selected = Signal(int, object)

    def __init__(self, image: QImage) -> None:
        super().__init__()
        self._image = image
        self._display_rect = QRectF()

        self._bounding_rect: QRectF | None = None
        self._corner_points: list[QPointF] = []

        self._is_dragging_bbox = False
        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None

        self._mouse_image_pos: QPointF | None = None
        self._mode = "bbox"

        self._corner_drag_index: int | None = None
        self._corner_drag_last_image_pos: QPointF | None = None
        self._corner_drag_sensitivity = 0.35
        self._corner_hit_radius_px = 14.0

        self.setMouseTracking(True)
        self.setMinimumSize(820, 520)

    @property
    def mode(self) -> str:
        """Return current interaction mode."""
        return self._mode

    @property
    def corner_points(self) -> list[QPointF]:
        """Return selected corner points in click order."""
        return list(self._corner_points)

    @property
    def bounding_rect(self) -> QRectF | None:
        """Return the currently selected image-space bounding rectangle, if set."""
        return self._bounding_rect

    def reset_bbox(self) -> None:
        """Reset all interaction state and restart bbox selection."""
        self._mode = "bbox"
        self._bounding_rect = None
        self._corner_points = []
        self._is_dragging_bbox = False
        self._drag_start = None
        self._drag_current = None
        self._corner_drag_index = None
        self._corner_drag_last_image_pos = None
        self.update()

    def reset_corners(self) -> None:
        """Clear corners while preserving selected bounding rectangle."""
        if self._bounding_rect is None:
            self.reset_bbox()
            return
        self._mode = "corner"
        self._corner_points = []
        self._corner_drag_index = None
        self._corner_drag_last_image_pos = None
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        """Paint image, overlays, and corner-selection magnifier."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        self._display_rect = self._compute_display_rect()
        if self._display_rect.isEmpty():
            return

        painter.drawImage(self._display_rect, self._image)
        self._draw_bounding_rect(painter)
        self._draw_corner_targets(painter)
        self._draw_magnifier(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Start bbox drag or place a corner marker."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        point = self._widget_to_image(event.position())
        if point is None:
            return

        self._mouse_image_pos = point

        if self._mode == "bbox":
            self._is_dragging_bbox = True
            self._drag_start = point
            self._drag_current = point
            self.update()
            return

        if self._mode == "corner":
            if self._bounding_rect is not None and not self._bounding_rect.contains(point):
                return
            drag_index = self._find_corner_index_near(event.position())
            if drag_index is not None:
                self._corner_drag_index = drag_index
                self._corner_drag_last_image_pos = point
                self.update()
                return
            if len(self._corner_points) >= 4:
                return
            self._corner_points.append(point)
            self.corner_selected.emit(len(self._corner_points), point)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Track pointer for bbox drag and magnifier updates."""
        point = self._widget_to_image(event.position())
        self._mouse_image_pos = point

        if self._mode == "bbox" and self._is_dragging_bbox and point is not None:
            self._drag_current = point

        if (
            self._mode == "corner"
            and self._corner_drag_index is not None
            and point is not None
            and self._corner_drag_last_image_pos is not None
        ):
            dx = point.x() - self._corner_drag_last_image_pos.x()
            dy = point.y() - self._corner_drag_last_image_pos.y()
            current = self._corner_points[self._corner_drag_index]

            # Dampen drag deltas so small hand movement enables finer corner control.
            moved = QPointF(
                current.x() + dx * self._corner_drag_sensitivity,
                current.y() + dy * self._corner_drag_sensitivity,
            )
            self._corner_points[self._corner_drag_index] = self._clamp_corner_to_bounds(moved)
            self._corner_drag_last_image_pos = point
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        """Commit bbox selection on drag release."""
        if self._mode != "bbox" or event.button() != Qt.MouseButton.LeftButton:
            if self._mode == "corner" and event.button() == Qt.MouseButton.LeftButton:
                self._corner_drag_index = None
                self._corner_drag_last_image_pos = None
            return
        if not self._is_dragging_bbox:
            return

        self._is_dragging_bbox = False
        if self._drag_start is None or self._drag_current is None:
            return

        rect = QRectF(self._drag_start, self._drag_current).normalized()
        self._drag_start = None
        self._drag_current = None

        if rect.width() < 8 or rect.height() < 8:
            self.update()
            return

        self._bounding_rect = rect
        self._mode = "corner"
        self.bounding_rect_defined.emit(rect)
        self.update()

    def leaveEvent(self, _event) -> None:  # type: ignore[override]
        """Clear pointer-dependent overlays when pointer leaves canvas."""
        self._mouse_image_pos = None
        self.update()

    def _compute_display_rect(self) -> QRectF:
        widget_width = float(self.width())
        widget_height = float(self.height())
        image_width = float(self._image.width())
        image_height = float(self._image.height())

        if widget_width <= 0 or widget_height <= 0 or image_width <= 0 or image_height <= 0:
            return QRectF()

        image_aspect = image_width / image_height
        widget_aspect = widget_width / widget_height

        if image_aspect > widget_aspect:
            draw_width = widget_width
            draw_height = draw_width / image_aspect
        else:
            draw_height = widget_height
            draw_width = draw_height * image_aspect

        return QRectF(
            (widget_width - draw_width) / 2.0,
            (widget_height - draw_height) / 2.0,
            draw_width,
            draw_height,
        )

    def _widget_to_image(self, point: QPointF) -> QPointF | None:
        if self._display_rect.isEmpty() or not self._display_rect.contains(point):
            return None

        nx = (point.x() - self._display_rect.x()) / self._display_rect.width()
        ny = (point.y() - self._display_rect.y()) / self._display_rect.height()
        return QPointF(
            nx * float(self._image.width() - 1),
            ny * float(self._image.height() - 1),
        )

    def _image_to_widget(self, point: QPointF) -> QPointF:
        return QPointF(
            self._display_rect.x()
            + (point.x() / float(max(1, self._image.width() - 1)))
            * self._display_rect.width(),
            self._display_rect.y()
            + (point.y() / float(max(1, self._image.height() - 1)))
            * self._display_rect.height(),
        )

    def _find_corner_index_near(self, widget_pos: QPointF) -> int | None:
        if not self._corner_points:
            return None

        for index, corner in enumerate(self._corner_points):
            handle = self._image_to_widget(corner)
            if (
                abs(handle.x() - widget_pos.x()) <= self._corner_hit_radius_px
                and abs(handle.y() - widget_pos.y()) <= self._corner_hit_radius_px
            ):
                return index
        return None

    def _clamp_corner_to_bounds(self, point: QPointF) -> QPointF:
        clamped_x = min(max(0.0, point.x()), float(self._image.width() - 1))
        clamped_y = min(max(0.0, point.y()), float(self._image.height() - 1))
        clamped = QPointF(clamped_x, clamped_y)

        if self._bounding_rect is None:
            return clamped

        rect = self._bounding_rect
        bounded_x = min(max(float(rect.left()), clamped.x()), float(rect.right()))
        bounded_y = min(max(float(rect.top()), clamped.y()), float(rect.bottom()))
        return QPointF(bounded_x, bounded_y)

    def _draw_bounding_rect(self, painter: QPainter) -> None:
        if self._bounding_rect is not None:
            tl = self._image_to_widget(self._bounding_rect.topLeft())
            br = self._image_to_widget(self._bounding_rect.bottomRight())
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2.0, Qt.PenStyle.SolidLine))
            painter.drawRect(QRectF(tl, br).normalized())

        if (
            self._is_dragging_bbox
            and self._drag_start is not None
            and self._drag_current is not None
        ):
            tl = self._image_to_widget(self._drag_start)
            br = self._image_to_widget(self._drag_current)
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2.0, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(tl, br).normalized())

    def _draw_corner_targets(self, painter: QPainter) -> None:
        if not self._corner_points:
            return

        painter.setPen(QPen(Qt.GlobalColor.red, 2.0, Qt.PenStyle.SolidLine))
        for index, corner in enumerate(self._corner_points, start=1):
            wpt = self._image_to_widget(corner)
            radius = 8.0
            painter.drawEllipse(wpt, radius, radius)
            painter.drawLine(
                QPointF(wpt.x() - 12.0, wpt.y()),
                QPointF(wpt.x() + 12.0, wpt.y()),
            )
            painter.drawLine(
                QPointF(wpt.x(), wpt.y() - 12.0),
                QPointF(wpt.x(), wpt.y() + 12.0),
            )
            painter.drawText(QPointF(wpt.x() + 10.0, wpt.y() - 10.0), str(index))

    def _draw_magnifier(self, painter: QPainter) -> None:
        if self._mode != "corner" or self._mouse_image_pos is None:
            return

        sample_half = 20
        cx = int(round(self._mouse_image_pos.x()))
        cy = int(round(self._mouse_image_pos.y()))

        left = max(0, cx - sample_half)
        top = max(0, cy - sample_half)
        right = min(self._image.width() - 1, cx + sample_half)
        bottom = min(self._image.height() - 1, cy + sample_half)

        crop = self._image.copy(
            left,
            top,
            max(1, right - left + 1),
            max(1, bottom - top + 1),
        )
        magnifier_size = 150
        magnified = crop.scaled(
            magnifier_size,
            magnifier_size,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        margin = 14
        dst_rect = QRectF(
            float(self.width() - magnifier_size - margin),
            float(margin),
            float(magnifier_size),
            float(magnifier_size),
        )

        painter.setPen(QPen(Qt.GlobalColor.white, 2.0, Qt.PenStyle.SolidLine))
        painter.drawRect(dst_rect)
        painter.drawImage(dst_rect, magnified)

        center = dst_rect.center()
        painter.setPen(QPen(Qt.GlobalColor.green, 1.5, Qt.PenStyle.SolidLine))
        painter.drawLine(
            QPointF(dst_rect.left(), center.y()),
            QPointF(dst_rect.right(), center.y()),
        )
        painter.drawLine(
            QPointF(center.x(), dst_rect.top()),
            QPointF(center.x(), dst_rect.bottom()),
        )


class StraightenImageDialog(QDialog):
    """Dialog guiding bbox selection and free-order corner picking."""

    def __init__(self, *, image_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Straighten Image")
        self.resize(980, 760)

        self._image_path = image_path
        self._straightened_image_path: Path | None = None

        image = QImage(str(image_path))
        if image.isNull():
            raise ValueError(f"Unable to load image: {image_path}")

        self.instructions_label = QLabel()
        self.instructions_label.setWordWrap(True)

        self.canvas = CornerSelectionCanvas(image)
        self.canvas.bounding_rect_defined.connect(self._on_bounding_rect_defined)
        self.canvas.corner_selected.connect(self._on_corner_selected)

        self.reset_bbox_button = QPushButton("Reset Bounding Area")
        self.reset_bbox_button.clicked.connect(self._reset_bbox)

        self.reset_corners_button = QPushButton("Reset Corners")
        self.reset_corners_button.clicked.connect(self._reset_corners)
        self.reset_corners_button.setEnabled(False)

        self.straighten_button = QPushButton("Straighten")
        self.straighten_button.clicked.connect(self._straighten_image)
        self.straighten_button.setEnabled(False)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        button_row = QHBoxLayout()
        button_row.addWidget(self.reset_bbox_button)
        button_row.addWidget(self.reset_corners_button)
        button_row.addStretch(1)
        button_row.addWidget(self.straighten_button)
        button_row.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.instructions_label)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(button_row)

        self._update_instructions()

    @property
    def straightened_image_path(self) -> Path | None:
        """Return the most recently saved straightened image path."""
        return self._straightened_image_path

    def _update_instructions(self) -> None:
        if self.canvas.mode == "bbox":
            self.instructions_label.setText(
                "Step 1: Drag a rectangle over the image area to work within."
            )
            return

        picked = len(self.canvas.corner_points)
        if picked < 4:
            self.instructions_label.setText(
                f"Step 2: Click 4 corners in any order. Selected: {picked}/4."
            )
            return

        self.instructions_label.setText(
            "All corners selected. Click Straighten to compute a rectified image."
        )

    def _on_bounding_rect_defined(self, _rect: QRectF) -> None:
        self.reset_corners_button.setEnabled(True)
        self._update_instructions()

    def _on_corner_selected(self, _index: int, _point: QPointF) -> None:
        self.straighten_button.setEnabled(len(self.canvas.corner_points) == 4)
        self._update_instructions()

    def _reset_bbox(self) -> None:
        self.canvas.reset_bbox()
        self.reset_corners_button.setEnabled(False)
        self.straighten_button.setEnabled(False)
        self._update_instructions()

    def _reset_corners(self) -> None:
        self.canvas.reset_corners()
        self.straighten_button.setEnabled(False)
        self._update_instructions()

    def _straighten_image(self) -> None:
        if len(self.canvas.corner_points) != 4:
            QMessageBox.warning(
                self,
                "Missing Corners",
                "Please select all four corners.",
            )
            return

        source: npt.NDArray[np.float32] = np.array(
            [[point.x(), point.y()] for point in self.canvas.corner_points],
            dtype=np.float32,
        )

        preserve_corners: npt.NDArray[np.float32] | None = None
        if self.canvas.bounding_rect is not None:
            rect = self.canvas.bounding_rect
            preserve_corners = np.array(
                [
                    [float(rect.left()), float(rect.top())],
                    [float(rect.right()), float(rect.top())],
                    [float(rect.right()), float(rect.bottom())],
                    [float(rect.left()), float(rect.bottom())],
                ],
                dtype=np.float32,
            )

        original_mat = cv2.imread(str(self._image_path), cv2.IMREAD_COLOR)
        if original_mat is None:
            QMessageBox.critical(
                self,
                "Load Error",
                "Unable to read source image for warp.",
            )
            return

        original: npt.NDArray[np.uint8] = np.ascontiguousarray(
            original_mat, dtype=np.uint8
        )

        try:
            result = straighten_full_frame(
                original,
                source,
                preserve_corners=preserve_corners,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Selection", str(exc))
            return

        out_path = self._image_path.with_name(
            f"{self._image_path.stem}_straightened{self._image_path.suffix}"
        )

        if not cv2.imwrite(str(out_path), result.warped_image):
            QMessageBox.critical(
                self,
                "Save Error",
                "Unable to write straightened image.",
            )
            return

        self._straightened_image_path = out_path

        QMessageBox.information(
            self,
            "Straightened Image Saved",
            f"Saved to:\n{out_path}",
        )
