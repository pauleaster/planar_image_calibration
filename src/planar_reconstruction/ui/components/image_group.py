"""Image preview panel for reconstruction outputs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class ImageGroup(QGroupBox):
    """Shows the latest reconstructed image and keeps it scaled to fit."""

    def __init__(self) -> None:
        super().__init__("Final Reconstruction")
        layout = QVBoxLayout(self)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.straighten_button = QPushButton("Straighten Image")
        self.straighten_button.setEnabled(False)
        button_row.addWidget(self.straighten_button)
        layout.addLayout(button_row)

        self.image_label = QLabel("No image yet")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(260)
        self.image_label.setStyleSheet(
            "border: 1px solid #999; background: #111; color: #eee;"
        )
        layout.addWidget(self.image_label)

        self._source_pixmap = QPixmap()
        self._current_image_path: Path | None = None

    @property
    def current_image_path(self) -> Path | None:
        """Return the image path currently shown in the preview, if any."""
        return self._current_image_path

    def clear(self, message: str) -> None:
        """Reset preview state and show a status message."""
        self._source_pixmap = QPixmap()
        self._current_image_path = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(message)
        self.straighten_button.setEnabled(False)

    def set_image(self, image_path: Path) -> bool:
        """Load and display an image path. Returns False on load failure."""
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.clear("Failed to load output image")
            return False

        self._source_pixmap = pixmap
        self._current_image_path = image_path
        self.straighten_button.setEnabled(True)
        self._render_scaled_pixmap()
        return True

    def refresh_scale(self) -> None:
        """Re-render the current image to match the current widget size."""
        if self._source_pixmap.isNull():
            return
        self._render_scaled_pixmap()

    def _render_scaled_pixmap(self) -> None:
        scaled = self._source_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setText("")
        self.image_label.setPixmap(scaled)
