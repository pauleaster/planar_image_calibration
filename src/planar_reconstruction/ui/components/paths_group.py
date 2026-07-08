"""Input and action controls for selecting paths and starting a run."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton


class PathsGroup(QGroupBox):
    """Reusable group for input/output path controls and run action."""

    def __init__(self, *, default_output_dir: Path) -> None:
        super().__init__("Inputs")
        grid = QGridLayout(self)

        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select input video file...")

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output directory...")
        self.output_edit.setText(str(default_output_dir))

        self.video_button = QPushButton("Select Video")
        self.output_button = QPushButton("Select Output Folder")
        self.run_button = QPushButton("Run")
        self.run_button.setMinimumHeight(38)
        self.run_button.setStyleSheet(
            """
            QPushButton {
                font-weight: 700;
                color: #ffffff;
                background-color: #1f6feb;
                border: 1px solid #1a5fcc;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #2b7bff;
            }
            QPushButton:pressed {
                background-color: #1757bb;
            }
            QPushButton:disabled {
                color: #d2d9e6;
                background-color: #6f7f98;
                border-color: #6f7f98;
            }
            """
        )

        grid.addWidget(QLabel("Video"), 0, 0)
        grid.addWidget(self.video_edit, 0, 1)
        grid.addWidget(self.video_button, 0, 2)

        grid.addWidget(QLabel("Output"), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(self.output_button, 1, 2)

        grid.addWidget(QLabel("Run"), 2, 0)
        grid.addWidget(self.run_button, 2, 1, 1, 2)

    @property
    def video_path_text(self) -> str:
        """Return the current raw video path text from the editor."""
        return self.video_edit.text().strip()

    @property
    def output_dir_text(self) -> str:
        """Return the current raw output base directory text from the editor."""
        return self.output_edit.text().strip()
