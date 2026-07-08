"""Run-configuration controls used by the reconstruction window."""

from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QGridLayout, QGroupBox, QLabel, QSpinBox


class OptionsGroup(QGroupBox):
    """Reusable group containing reconstruction threshold and sampling options."""

    def __init__(self) -> None:
        """Initialize spin-box controls and defaults for one reconstruction run."""
        super().__init__("Run Options")
        grid = QGridLayout(self)

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 500)
        self.frame_step_spin.setValue(5)

        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(1, 100000)
        self.max_frames_spin.setValue(300)

        self.min_sharpness_spin = QDoubleSpinBox()
        self.min_sharpness_spin.setRange(0.0, 10000.0)
        self.min_sharpness_spin.setDecimals(2)
        self.min_sharpness_spin.setValue(100.0)

        self.min_inliers_spin = QSpinBox()
        self.min_inliers_spin.setRange(1, 10000)
        self.min_inliers_spin.setValue(4)

        self.min_inlier_ratio_spin = QDoubleSpinBox()
        self.min_inlier_ratio_spin.setRange(0.0, 1.0)
        self.min_inlier_ratio_spin.setSingleStep(0.05)
        self.min_inlier_ratio_spin.setDecimals(3)
        self.min_inlier_ratio_spin.setValue(0.5)

        grid.addWidget(QLabel("Frame Step"), 0, 0)
        grid.addWidget(self.frame_step_spin, 0, 1)
        grid.addWidget(QLabel("Max Frames"), 0, 2)
        grid.addWidget(self.max_frames_spin, 0, 3)

        grid.addWidget(QLabel("Min Sharpness"), 1, 0)
        grid.addWidget(self.min_sharpness_spin, 1, 1)
        grid.addWidget(QLabel("Min Inliers"), 1, 2)
        grid.addWidget(self.min_inliers_spin, 1, 3)

        grid.addWidget(QLabel("Min Inlier Ratio"), 2, 0)
        grid.addWidget(self.min_inlier_ratio_spin, 2, 1)

    @property
    def frame_step(self) -> int:
        """Return the current frame-step sampling value."""
        return self.frame_step_spin.value()

    @property
    def max_frames(self) -> int:
        """Return the maximum number of frames to process for the run."""
        return self.max_frames_spin.value()

    @property
    def min_sharpness(self) -> float:
        """Return the sharpness threshold for accepting a reference/frame."""
        return self.min_sharpness_spin.value()

    @property
    def min_inliers(self) -> int:
        """Return the minimum inlier-count threshold for homography acceptance."""
        return self.min_inliers_spin.value()

    @property
    def min_inlier_ratio(self) -> float:
        """Return the minimum inlier-ratio threshold for homography acceptance."""
        return self.min_inlier_ratio_spin.value()
