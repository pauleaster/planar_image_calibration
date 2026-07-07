"""PySide6 application entry point for the Planar Image Calibration Workbench."""

from __future__ import annotations

import sys
from typing import Sequence

from PySide6.QtWidgets import QApplication

from planar_reconstruction.ui.main_window import MainWindow


def main(argv: Sequence[str] | None = None) -> int:
    """Start the desktop UI and run the Qt event loop."""
    args = list(argv) if argv is not None else sys.argv
    app = QApplication(args)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
