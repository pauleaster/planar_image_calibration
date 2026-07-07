"""Diagnostics and summary helpers for planar reconstruction runs.

The reconstruction pipeline can use these utilities to collect run-level metrics
and write compact JSON summaries without depending on the CLI or UI layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RunSummary:
    """Compact metrics summary for a reconstruction run."""

    frames_read: int
    frames_processed: int
    frames_accepted: int
    frames_rejected: int
    mean_sharpness: float
    mean_inlier_ratio: float
    reference_frame_saved: bool
    reference_frame_path: str | None
    output_image: str | None = None
    diagnostics: list[Mapping[str, object]] = field(default_factory=list)


def build_run_summary(
    *,
    frames_read: int,
    frames_processed: int,
    frames_accepted: int,
    frames_rejected: int,
    mean_sharpness: float,
    mean_inlier_ratio: float,
    reference_frame_saved: bool,
    reference_frame_path: str | Path | None,
    output_image: str | Path | None = None,
    diagnostics: Sequence[Mapping[str, object]] | None = None,
) -> RunSummary:
    """Build a summary object from primitive run metrics."""
    return RunSummary(
        frames_read=frames_read,
        frames_processed=frames_processed,
        frames_accepted=frames_accepted,
        frames_rejected=frames_rejected,
        mean_sharpness=mean_sharpness,
        mean_inlier_ratio=mean_inlier_ratio,
        reference_frame_saved=reference_frame_saved,
        reference_frame_path=str(reference_frame_path) if reference_frame_path is not None else None,
        output_image=str(output_image) if output_image is not None else None,
        diagnostics=list(diagnostics or []),
    )


def summary_to_dict(summary: RunSummary) -> dict[str, object]:
    """Convert a summary object into a JSON-serializable dictionary."""
    return {
        "frames_read": summary.frames_read,
        "frames_processed": summary.frames_processed,
        "frames_accepted": summary.frames_accepted,
        "frames_rejected": summary.frames_rejected,
        "mean_sharpness": summary.mean_sharpness,
        "mean_inlier_ratio": summary.mean_inlier_ratio,
        "reference_frame_saved": summary.reference_frame_saved,
        "reference_frame_path": summary.reference_frame_path,
        "output_image": summary.output_image,
        "diagnostics": list(summary.diagnostics),
    }


def write_summary_json(summary_path: str | Path, summary: RunSummary) -> None:
    """Write a summary JSON file to disk."""
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_to_dict(summary), indent=2), encoding="utf-8")
