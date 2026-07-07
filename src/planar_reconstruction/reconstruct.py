"""Reconstruction orchestration for the planar image calibration workflow.

This module is intentionally UI-agnostic and CLI-agnostic. It ties together the
stream, quality, feature, and homography helpers so the project has a single
core pipeline entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2

from planar_reconstruction.features import FeatureSet, detect_features, match_features
from planar_reconstruction.homography import estimate_homography
from planar_reconstruction.quality import compute_sharpness_score
from planar_reconstruction.stream import FramePacket


@dataclass(frozen=True)
class FrameDiagnostics:
    """Diagnostics collected for one processed frame."""

    index: int
    timestamp_ms: float
    sharpness_score: float
    accepted: bool
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0


@dataclass(frozen=True)
class ReconstructionResult:
    """Summary of a reconstruction run and its primary output paths."""

    frames_read: int
    frames_processed: int
    frames_accepted: int
    frames_rejected: int
    mean_sharpness: float
    mean_inlier_ratio: float
    reference_frame_saved: bool
    reference_frame_path: Path | None
    summary_path: Path | None
    diagnostics: list[FrameDiagnostics] = field(default_factory=list)


@dataclass(frozen=True)
class ReconstructionOptions:
    """Configuration for the initial reconstruction pipeline."""

    output_dir: Path
    min_sharpness: float = 100.0
    min_inliers: int = 4
    min_inlier_ratio: float = 0.5
    max_features: int = 1000
    ratio_threshold: float = 0.75
    ransac_reproj_threshold: float = 5.0



def reconstruct_frames(
    frame_packets: Iterable[FramePacket],
    options: ReconstructionOptions,
) -> ReconstructionResult:
    """Process a frame stream and write the first useful reference frame.

    The current version is intentionally small: it measures sharpness, stores the
    first sufficiently sharp frame as the reference image, and records basic
    per-frame diagnostics. Feature matching and homography estimation are wired
    in when enough information is available, but they do not gate output yet.
    """
    options.output_dir.mkdir(parents=True, exist_ok=True)

    frames_read = 0
    frames_processed = 0
    frames_accepted = 0
    frames_rejected = 0
    sharpness_values: list[float] = []
    inlier_ratios: list[float] = []
    diagnostics: list[FrameDiagnostics] = []
    reference_frame_saved = False
    reference_frame_path = options.output_dir / "reference_frame.png"

    reference_features: FeatureSet | None = None

    for packet in frame_packets:
        frames_read += 1
        frames_processed += 1

        sharpness_score = compute_sharpness_score(packet.frame)
        sharpness_values.append(sharpness_score)

        accepted = sharpness_score >= options.min_sharpness
        match_count = 0
        inlier_count = 0
        inlier_ratio = 0.0

        if accepted and not reference_frame_saved:
            cv2.imwrite(str(reference_frame_path), packet.frame)  # pylint: disable=no-member
            reference_frame_saved = True
            reference_features = detect_features(packet.frame, max_features=options.max_features)
            frames_accepted += 1
        elif accepted and reference_features is not None:
            current_features = detect_features(packet.frame, max_features=options.max_features)
            match_result = match_features(
                reference_features,
                current_features,
                ratio_threshold=options.ratio_threshold,
            )
            match_count = len(match_result.matches)

            if match_result.source_points.shape[0] >= 4:
                homography_result = estimate_homography(
                    match_result.source_points,
                    match_result.target_points,
                    ransac_reproj_threshold=options.ransac_reproj_threshold,
                )
                inlier_count = homography_result.inlier_count
                inlier_ratio = homography_result.inlier_ratio
                inlier_ratios.append(inlier_ratio)

                if homography_result.matrix is not None and (
                    inlier_count >= options.min_inliers
                    and inlier_ratio >= options.min_inlier_ratio
                ):
                    frames_accepted += 1
                else:
                    frames_rejected += 1
            else:
                frames_rejected += 1
        else:
            frames_rejected += 1

        diagnostics.append(
            FrameDiagnostics(
                index=packet.index,
                timestamp_ms=packet.timestamp_ms,
                sharpness_score=sharpness_score,
                accepted=accepted,
                match_count=match_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
            )
        )

    mean_sharpness = float(sum(sharpness_values) / len(sharpness_values)) if sharpness_values else 0.0
    mean_inlier_ratio = float(sum(inlier_ratios) / len(inlier_ratios)) if inlier_ratios else 0.0
    summary_path = options.output_dir / "summary.json"

    summary = ReconstructionResult(
        frames_read=frames_read,
        frames_processed=frames_processed,
        frames_accepted=frames_accepted,
        frames_rejected=frames_rejected,
        mean_sharpness=mean_sharpness,
        mean_inlier_ratio=mean_inlier_ratio,
        reference_frame_saved=reference_frame_saved,
        reference_frame_path=reference_frame_path if reference_frame_saved else None,
        summary_path=summary_path,
        diagnostics=diagnostics,
    )

    _write_summary_json(summary)
    return summary



def _write_summary_json(summary: ReconstructionResult) -> None:
    """Write a compact summary JSON next to the output images."""
    data = {
        "frames_read": summary.frames_read,
        "frames_processed": summary.frames_processed,
        "frames_accepted": summary.frames_accepted,
        "frames_rejected": summary.frames_rejected,
        "mean_sharpness": summary.mean_sharpness,
        "mean_inlier_ratio": summary.mean_inlier_ratio,
        "reference_frame_saved": summary.reference_frame_saved,
        "reference_frame_path": str(summary.reference_frame_path) if summary.reference_frame_path else None,
    }
    if summary.summary_path is not None:
        summary.summary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
