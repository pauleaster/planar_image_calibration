"""Status output section for user-facing progress text."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout


class StatusGroup(QGroupBox):
    """Reusable status section with a wrapped single-line label."""

    def __init__(self) -> None:
        super().__init__("Status")
        layout = QVBoxLayout(self)
        self.status_label = QLabel(
            "Ready. Select a video and output folder, then click 'Run'."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def set_status(self, text: str) -> None:
        """Set the status message shown to the user."""
        self.status_label.setText(text)
