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
import numpy as np

from planar_reconstruction.features import FeatureSet, detect_features, match_features
from planar_reconstruction.homography import estimate_homography
from planar_reconstruction.quality import compute_sharpness_score
from planar_reconstruction.stream import FramePacket

def _empty_sharpness_scores() -> list[float]:
    return []

def _empty_frame_diagnostics() -> list[FrameDiagnostics]:
    return []


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
    output_image_path: Path | None
    fused_frame_count: int
    summary_path: Path | None
    diagnostics_summary_path: Path | None
    debug_images_dir: Path | None
    sharpness_scores: list[float] = field(default_factory=_empty_sharpness_scores)
    diagnostics: list[FrameDiagnostics] = field(default_factory=_empty_frame_diagnostics)


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
    save_debug_images: bool = False
    debug_image_limit: int = 100


def reconstruct_frames(
    frame_packets: Iterable[FramePacket],
    options: ReconstructionOptions,
) -> ReconstructionResult:
    """Process a frame stream and write the first useful reference frame.

    The pipeline tracks per-frame diagnostics, writes a compact run summary, and
    writes a detailed diagnostics summary JSON with threshold values and
    sharpness traces. Optional debug image dumps can be enabled for inspection.
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
    output_image_path = options.output_dir / "final_reconstruction.png"
    diagnostics_summary_path = options.output_dir / "diagnostics_summary.json"
    debug_images_dir = (
        options.output_dir / "debug_images" if options.save_debug_images else None
    )
    debug_images_written = 0

    fused_accumulator: np.ndarray | None = None
    fused_frame_count = 0
    reference_size: tuple[int, int] | None = None

    reference_features: FeatureSet | None = None

    for packet in frame_packets:
        frames_read += 1
        frames_processed += 1

        sharpness_score = compute_sharpness_score(packet.frame)
        sharpness_values.append(sharpness_score)

        sharpness_ok = sharpness_score >= options.min_sharpness
        accepted = False
        match_count = 0
        inlier_count = 0
        inlier_ratio = 0.0

        if sharpness_ok and not reference_frame_saved:
            cv2.imwrite(str(reference_frame_path), packet.frame)
            reference_frame_saved = True
            reference_features = detect_features(
                packet.frame, max_features=options.max_features
            )
            reference_size = (packet.frame.shape[1], packet.frame.shape[0])
            fused_accumulator = packet.frame.astype(np.float32)
            fused_frame_count = 1
            accepted = True
        elif (
            sharpness_ok
            and reference_features is not None
            and reference_size is not None
        ):
            current_features = detect_features(
                packet.frame, max_features=options.max_features
            )
            match_result = match_features(
                current_features,
                reference_features,
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
                    warped = cv2.warpPerspective(
                        packet.frame,
                        homography_result.matrix,
                        reference_size,
                    )
                    if fused_accumulator is None:
                        fused_accumulator = warped.astype(np.float32)
                        fused_frame_count = 1
                    else:
                        fused_accumulator += warped.astype(np.float32)
                        fused_frame_count += 1
                    accepted = True

        if accepted:
            frames_accepted += 1
        else:
            frames_rejected += 1

        if (
            options.save_debug_images
            and debug_images_dir is not None
            and debug_images_written < options.debug_image_limit
        ):
            debug_images_dir.mkdir(parents=True, exist_ok=True)
            label = "accepted" if accepted else "rejected"
            debug_image_path = debug_images_dir / f"{packet.index:06d}_{label}.png"
            cv2.imwrite(str(debug_image_path), packet.frame)
            debug_images_written += 1

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

    mean_sharpness = (
        float(sum(sharpness_values) / len(sharpness_values))
        if sharpness_values
        else 0.0
    )
    mean_inlier_ratio = (
        float(sum(inlier_ratios) / len(inlier_ratios)) if inlier_ratios else 0.0
    )
    summary_path = options.output_dir / "summary.json"

    saved_output_image: Path | None = None
    if fused_accumulator is not None and fused_frame_count > 0:
        fused_image = np.clip(
            fused_accumulator / float(fused_frame_count), 0, 255
        ).astype(np.uint8)
        cv2.imwrite(str(output_image_path), fused_image)
        saved_output_image = output_image_path

    summary = ReconstructionResult(
        frames_read=frames_read,
        frames_processed=frames_processed,
        frames_accepted=frames_accepted,
        frames_rejected=frames_rejected,
        mean_sharpness=mean_sharpness,
        mean_inlier_ratio=mean_inlier_ratio,
        reference_frame_saved=reference_frame_saved,
        reference_frame_path=reference_frame_path if reference_frame_saved else None,
        output_image_path=saved_output_image,
        fused_frame_count=fused_frame_count,
        summary_path=summary_path,
        diagnostics_summary_path=diagnostics_summary_path,
        debug_images_dir=debug_images_dir,
        sharpness_scores=sharpness_values,
        diagnostics=diagnostics,
    )

    _write_summary_json(summary)
    _write_diagnostics_summary_json(summary, options, debug_images_written)
    return summary


def _write_summary_json(summary: ReconstructionResult) -> None:
    """Write a compact summary JSON next to the output images."""
    data: dict[str, object] = {
        "frames_read": summary.frames_read,
        "frames_processed": summary.frames_processed,
        "frames_accepted": summary.frames_accepted,
        "frames_rejected": summary.frames_rejected,
        "mean_sharpness": summary.mean_sharpness,
        "mean_inlier_ratio": summary.mean_inlier_ratio,
        "reference_frame_saved": summary.reference_frame_saved,
        "reference_frame_path": (
            str(summary.reference_frame_path) if summary.reference_frame_path else None
        ),
        "output_image": (
            str(summary.output_image_path) if summary.output_image_path else None
        ),
        "fused_frame_count": summary.fused_frame_count,
    }
    if summary.summary_path is not None:
        summary.summary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )


def _write_diagnostics_summary_json(
    summary: ReconstructionResult,
    options: ReconstructionOptions,
    debug_images_written: int,
) -> None:
    """Write detailed diagnostics, traces, and thresholds for one run."""
    data: dict[str, object] = {
        "frames_read": summary.frames_read,
        "frames_processed": summary.frames_processed,
        "frames_accepted": summary.frames_accepted,
        "frames_rejected": summary.frames_rejected,
        "mean_sharpness": summary.mean_sharpness,
        "mean_inlier_ratio": summary.mean_inlier_ratio,
        "sharpness_scores": summary.sharpness_scores,
        "thresholds": {
            "min_sharpness": options.min_sharpness,
            "min_inliers": options.min_inliers,
            "min_inlier_ratio": options.min_inlier_ratio,
        },
        "reference_frame_saved": summary.reference_frame_saved,
        "reference_frame_path": (
            str(summary.reference_frame_path) if summary.reference_frame_path else None
        ),
        "output_image": (
            str(summary.output_image_path) if summary.output_image_path else None
        ),
        "fused_frame_count": summary.fused_frame_count,
        "debug_images": {
            "enabled": options.save_debug_images,
            "directory": (
                str(summary.debug_images_dir) if summary.debug_images_dir else None
            ),
            "written": debug_images_written,
            "limit": options.debug_image_limit,
        },
        "per_frame": [
            {
                "index": frame.index,
                "timestamp_ms": frame.timestamp_ms,
                "sharpness_score": frame.sharpness_score,
                "accepted": frame.accepted,
                "match_count": frame.match_count,
                "inlier_count": frame.inlier_count,
                "inlier_ratio": frame.inlier_ratio,
            }
            for frame in summary.diagnostics
        ],
    }

    if summary.diagnostics_summary_path is not None:
        summary.diagnostics_summary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
