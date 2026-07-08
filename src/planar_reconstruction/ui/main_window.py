"""Main PySide6 window for running planar reconstruction."""

from __future__ import annotations

from datetime import datetime
from functools import partial
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from planar_reconstruction.reconstruct import (
    ReconstructionOptions,
    ReconstructionResult,
    reconstruct_frames,
)
from planar_reconstruction.stream import iter_video_frames
from planar_reconstruction.ui.straighten_dialog import StraightenImageDialog
from planar_reconstruction.ui.components import (
    ImageGroup,
    OptionsGroup,
    PathsGroup,
    StatusGroup,
    SummaryGroup,
)


def _build_run_output_dir(base_output_dir: Path) -> Path:
    """Create a unique timestamped output directory for one run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_output_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = base_output_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


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

        self._default_input_dir = Path("./data/input")
        self._default_output_dir = Path("./data/output")

        self._thread: QThread | None = None
        self._worker: ReconstructionWorker | None = None
        self._straighten_dialogs: list[StraightenImageDialog] = []

        self.paths_group = PathsGroup(default_output_dir=self._default_output_dir)
        self.options_group = OptionsGroup()
        self.status_group = StatusGroup()
        self.image_group = ImageGroup()
        self.summary_group = SummaryGroup()

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.addWidget(self.paths_group)
        layout.addWidget(self.options_group)
        layout.addWidget(self.status_group)
        layout.addWidget(self.image_group)
        layout.addWidget(self.summary_group)
        self.setCentralWidget(root)

        self.paths_group.video_button.clicked.connect(self._select_video)  # pylint: disable=no-member
        self.paths_group.output_button.clicked.connect(self._select_output_dir)  # pylint: disable=no-member
        self.paths_group.run_button.clicked.connect(self._run_reconstruction)  # pylint: disable=no-member
        self.image_group.straighten_button.clicked.connect(self._open_straighten_dialog)  # pylint: disable=no-member

    @Slot()
    def _select_video(self) -> None:
        start_dir = (
            self._default_input_dir if self._default_input_dir.exists() else Path.cwd()
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            str(start_dir),
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)",
        )
        if path:
            self.paths_group.video_edit.setText(path)

    @Slot()
    def _select_output_dir(self) -> None:
        current_output = self.paths_group.output_dir_text
        start_dir = Path(current_output) if current_output else self._default_output_dir
        if not start_dir.exists():
            start_dir = Path.cwd()
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(start_dir)
        )
        if path:
            self.paths_group.output_edit.setText(path)

    @Slot()
    def _run_reconstruction(self) -> None:
        video_path = Path(self.paths_group.video_path_text)
        output_text = self.paths_group.output_dir_text

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
        base_output_dir = Path(output_text)
        base_output_dir.mkdir(parents=True, exist_ok=True)
        run_output_dir = _build_run_output_dir(base_output_dir)

        self.paths_group.run_button.setEnabled(False)
        self.status_group.set_status(
            f"Starting background worker in {run_output_dir}..."
        )
        self.summary_group.clear()

        self._thread = QThread(self)
        self._worker = ReconstructionWorker(
            video_path=video_path,
            output_dir=run_output_dir,
            frame_step=self.options_group.frame_step,
            max_frames=self.options_group.max_frames,
            min_sharpness=self.options_group.min_sharpness,
            min_inliers=self.options_group.min_inliers,
            min_inlier_ratio=self.options_group.min_inlier_ratio,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)  # pylint: disable=no-member
        self._worker.progress.connect(self._set_status)  # pylint: disable=no-member
        self._worker.succeeded.connect(self._handle_success)  # pylint: disable=no-member
        self._worker.failed.connect(self._handle_error)  # pylint: disable=no-member
        self._worker.finished.connect(self._cleanup_thread)  # pylint: disable=no-member
        self._worker.finished.connect(self._thread.quit)  # pylint: disable=no-member

        self._thread.start()

    @Slot(str)
    def _set_status(self, text: str) -> None:
        self.status_group.set_status(text)

    @Slot(object)
    def _handle_success(self, result_obj: object) -> None:
        result = result_obj
        if not isinstance(result, ReconstructionResult):
            self._handle_error("Unexpected reconstruction result type.")
            return

        self.status_group.set_status("Reconstruction completed successfully.")
        self.summary_group.set_summary(self._format_summary_text(result))

        image_path = result.output_image_path or result.reference_frame_path
        if image_path is None or not image_path.exists():
            self.image_group.clear("No output image generated")
            return

        self.image_group.set_image(image_path)

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        self.status_group.set_status("Reconstruction failed.")
        QMessageBox.critical(self, "Reconstruction Error", message)

    @Slot()
    def _open_straighten_dialog(self) -> None:
        image_path = self.image_group.current_image_path
        if image_path is None or not image_path.exists():
            QMessageBox.warning(
                self,
                "Missing Image",
                "No generated output image is available to straighten.",
            )
            return

        try:
            dialog = StraightenImageDialog(image_path=image_path, parent=self)
        except ValueError as exc:
            QMessageBox.critical(self, "Open Error", str(exc))
            return

        dialog.finished.connect(partial(self._on_straighten_dialog_finished, dialog))  # pylint: disable=no-member
        self._straighten_dialogs.append(dialog)
        dialog.show()

    def _on_straighten_dialog_finished(
        self, dialog: StraightenImageDialog, _result: int
    ) -> None:
        """Handle straighten dialog closure and release retained references."""
        straightened_path = dialog.straightened_image_path
        if straightened_path is not None and straightened_path.exists():
            self.image_group.set_image(straightened_path)
            self.status_group.set_status(
                f"Loaded straightened image: {straightened_path.name}"
            )
        self._discard_straighten_dialog(dialog)

    def _discard_straighten_dialog(self, dialog: StraightenImageDialog) -> None:
        """Remove closed straighten dialog references."""
        self._straighten_dialogs = [
            item for item in self._straighten_dialogs if item is not dialog
        ]

    @Slot()
    def _cleanup_thread(self) -> None:
        self.paths_group.run_button.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep the preview image scaled when the window size changes."""
        super().resizeEvent(event)
        self.image_group.refresh_scale()

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
