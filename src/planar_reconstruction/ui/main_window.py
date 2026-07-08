"""Main PySide6 window for running planar reconstruction."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from planar_reconstruction.reconstruct import (
    ReconstructionOptions,
    ReconstructionResult,
    reconstruct_frames,
)
from planar_reconstruction.stream import iter_video_frames


class ReconstructionWorker(QObject):
    """Run reconstruction in a worker thread so the UI stays responsive."""

    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        *,
        video_path: Path,
        output_dir: Path,
        frame_step: int,
        max_frames: int,
        min_sharpness: float,
        min_inliers: int,
        min_inlier_ratio: float,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._output_dir = output_dir
        self._frame_step = frame_step
        self._max_frames = max_frames
        self._min_sharpness = min_sharpness
        self._min_inliers = min_inliers
        self._min_inlier_ratio = min_inlier_ratio

    @Slot()
    def run(self) -> None:
        """Execute frame streaming and reconstruction work."""
        try:
            self.progress.emit("Reading video frame stream...")
            packets = list(
                iter_video_frames(
                    self._video_path,
                    frame_step=self._frame_step,
                    max_frames=self._max_frames,
                )
            )
            self.progress.emit(
                f"Loaded {len(packets)} frames. Running reconstruction..."
            )

            options = ReconstructionOptions(
                output_dir=self._output_dir,
                min_sharpness=self._min_sharpness,
                min_inliers=self._min_inliers,
                min_inlier_ratio=self._min_inlier_ratio,
            )
            result = reconstruct_frames(packets, options)
            self.progress.emit("Reconstruction complete.")
            self.succeeded.emit(result)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class MainWindow(QMainWindow):
    """Minimal operator window for selecting inputs and running the pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Planar Image Calibration Workbench")
        self.resize(980, 720)

        self._thread: QThread | None = None
        self._worker: ReconstructionWorker | None = None

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_options_group())
        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_image_group())
        layout.addWidget(self._build_summary_group())
        self.setCentralWidget(root)

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Inputs")
        grid = QGridLayout(group)

        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select input video file...")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output directory...")

        self.video_button = QPushButton("Select Video")
        self.video_button.clicked.connect(self._select_video) # pylint: disable=no-member
        self.output_button = QPushButton("Select Output Folder")
        self.output_button.clicked.connect(self._select_output_dir) # pylint: disable=no-member
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._run_reconstruction) # pylint: disable=no-member
        self.run_button.setMinimumHeight(38)
        self.run_button.setStyleSheet("font-weight: 700;")

        grid.addWidget(QLabel("Video"), 0, 0)
        grid.addWidget(self.video_edit, 0, 1)
        grid.addWidget(self.video_button, 0, 2)

        grid.addWidget(QLabel("Output"), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(self.output_button, 1, 2)

        grid.addWidget(QLabel("Run"), 2, 0)
        grid.addWidget(self.run_button, 2, 1, 1, 2)
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Run Options")
        grid = QGridLayout(group)

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
        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)
        self.status_label = QLabel(
            "Ready. Select a video and output folder, then click 'Run'."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return group

    def _build_image_group(self) -> QGroupBox:
        group = QGroupBox("Final Reconstruction")
        layout = QVBoxLayout(group)
        self.image_label = QLabel("No image yet")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(260)
        self.image_label.setStyleSheet(
            "border: 1px solid #999; background: #111; color: #eee;"
        )
        layout.addWidget(self.image_label)
        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("Summary Metrics")
        layout = QVBoxLayout(group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        return group

    @Slot()
    def _select_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            str(Path.cwd()),
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)",
        )
        if path:
            self.video_edit.setText(path)

    @Slot()
    def _select_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(Path.cwd())
        )
        if path:
            self.output_edit.setText(path)

    @Slot()
    def _run_reconstruction(self) -> None:
        video_path = Path(self.video_edit.text().strip())
        output_text = self.output_edit.text().strip()

        if not video_path.exists():
            QMessageBox.warning(
                self, "Missing Input", "Please select a valid video file."
            )
            return
        if not output_text:
            QMessageBox.warning(
                self, "Missing Output", "Please select an output directory."
            )
            return
        output_dir = Path(output_text)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.run_button.setEnabled(False)
        self.status_label.setText("Starting background worker...")
        self.summary_text.clear()

        self._thread = QThread(self)
        self._worker = ReconstructionWorker(
            video_path=video_path,
            output_dir=output_dir,
            frame_step=self.frame_step_spin.value(),
            max_frames=self.max_frames_spin.value(),
            min_sharpness=self.min_sharpness_spin.value(),
            min_inliers=self.min_inliers_spin.value(),
            min_inlier_ratio=self.min_inlier_ratio_spin.value(),
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run) # pylint: disable=no-member
        self._worker.progress.connect(self._set_status)
        self._worker.succeeded.connect(self._handle_success)
        self._worker.failed.connect(self._handle_error)
        self._worker.finished.connect(self._cleanup_thread)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()

    @Slot(str)
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    @Slot(object)
    def _handle_success(self, result_obj: object) -> None:
        result = result_obj
        if not isinstance(result, ReconstructionResult):
            self._handle_error("Unexpected reconstruction result type.")
            return

        self.status_label.setText("Reconstruction completed successfully.")
        self.summary_text.setPlainText(self._format_summary_text(result))

        image_path = result.output_image_path or result.reference_frame_path
        if image_path is None or not image_path.exists():
            self.image_label.setText("No output image generated")
            self.image_label.setPixmap(QPixmap())
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.image_label.setText("Failed to load output image")
            self.image_label.setPixmap(QPixmap())
            return

        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        self.status_label.setText("Reconstruction failed.")
        QMessageBox.critical(self, "Reconstruction Error", message)

    @Slot()
    def _cleanup_thread(self) -> None:
        self.run_button.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def resize_event(self, event: QResizeEvent) -> None:
        """Keep the preview image scaled when the window size changes."""
        super().resizeEvent(event)
        pixmap = self.image_label.pixmap()
        if not pixmap.isNull():
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    @staticmethod
    def _format_summary_text(result: ReconstructionResult) -> str:
        return "\n".join(
            [
                f"Frames read: {result.frames_read}",
                f"Frames processed: {result.frames_processed}",
                f"Frames accepted: {result.frames_accepted}",
                f"Frames rejected: {result.frames_rejected}",
                f"Mean sharpness: {result.mean_sharpness:.2f}",
                f"Mean inlier ratio: {result.mean_inlier_ratio:.3f}",
                f"Reference frame: {result.reference_frame_path}",
                f"Final image: {result.output_image_path}",
                f"Summary: {result.summary_path}",
                f"Diagnostics: {result.diagnostics_summary_path}",
            ]
        )
