"""Summary metrics panel used by the reconstruction UI."""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QTextEdit, QVBoxLayout


class SummaryGroup(QGroupBox):
    """Read-only text area for showing run result summaries."""

    def __init__(self) -> None:
        super().__init__("Summary Metrics")
        layout = QVBoxLayout(self)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

    def clear(self) -> None:
        """Clear summary text content."""
        self.summary_text.clear()

    def set_summary(self, text: str) -> None:
        """Replace the summary text content."""
        self.summary_text.setPlainText(text)
