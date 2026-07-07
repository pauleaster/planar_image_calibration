"""Command-line interface for the planar reconstruction workflow.

The initial CLI focuses on argument parsing and validation so the project has a
stable Windows entry point before the frame-processing pipeline is added.
"""

# pylint: disable=no-member

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
from typing import Any, cast

import cv2

from planar_reconstruction.quality import compute_sharpness_score

CV2 = cast(Any, cv2)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the initial reconstruction CLI."""
    parser = argparse.ArgumentParser(
        prog="planar_reconstruction.cli",
        description=(
            "Run the planar image calibration workflow against a local video "
            "file on Windows."
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the input video file, for example data\\input\\painting_video.mp4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where outputs such as summary.json will be written.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Process every Nth frame. Default: 5.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to read from the video. Default: 300.",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=100.0,
        help="Minimum sharpness score for selecting the reference frame. Default: 100.0.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before the processing pipeline is invoked."""
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be greater than zero.")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero.")
    if args.min_sharpness < 0:
        raise ValueError("--min-sharpness must be non-negative.")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the initial reconstruction workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    capture = CV2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error(f"Could not open video: {args.video}")

    frames_read = 0
    frames_processed = 0
    frames_accepted = 0
    frames_rejected = 0
    processed_sharpness_scores: list[float] = []
    reference_frame_path = args.output_dir / "reference_frame.png"
    reference_frame_saved = False

    while frames_read < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break

        frame_index = frames_read
        frames_read += 1

        if frame_index % args.frame_step != 0:
            continue

        frames_processed += 1
        sharpness = compute_sharpness_score(frame)
        processed_sharpness_scores.append(sharpness)

        if sharpness >= args.min_sharpness:
            frames_accepted += 1
            if not reference_frame_saved:
                CV2.imwrite(str(reference_frame_path), frame)
                reference_frame_saved = True
        else:
            frames_rejected += 1

    capture.release()

    mean_sharpness = (
        float(sum(processed_sharpness_scores) / len(processed_sharpness_scores))
        if processed_sharpness_scores
        else 0.0
    )

    summary = {
        "video_path": str(args.video),
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "frames_accepted": frames_accepted,
        "frames_rejected": frames_rejected,
        "min_sharpness": args.min_sharpness,
        "mean_sharpness": mean_sharpness,
        "reference_frame_saved": reference_frame_saved,
        "reference_frame_path": str(reference_frame_path) if reference_frame_saved else None,
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Video input: {args.video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Frames read: {frames_read}")
    print(f"Frames processed: {frames_processed}")
    print(f"Frames accepted: {frames_accepted}")
    print(f"Frames rejected: {frames_rejected}")
    print(f"Reference frame saved: {reference_frame_saved}")
    print(f"Summary written: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
