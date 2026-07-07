"""Command-line interface for the planar reconstruction workflow.

The CLI is intentionally thin: it parses arguments, creates the frame stream,
and delegates all processing to the canonical reconstruction pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from planar_reconstruction.reconstruct import ReconstructionOptions, reconstruct_frames
from planar_reconstruction.stream import iter_video_frames


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
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=4,
        help="Minimum inlier count required to accept a registration. Default: 4.",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.5,
        help="Minimum inlier ratio required to accept a registration. Default: 0.5.",
    )
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help="Save per-frame debug images under output_dir/debug_images.",
    )
    parser.add_argument(
        "--debug-image-limit",
        type=int,
        default=100,
        help="Maximum number of debug images to write when enabled. Default: 100.",
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
    if args.min_inliers <= 0:
        raise ValueError("--min-inliers must be greater than zero.")
    if not 0.0 <= args.min_inlier_ratio <= 1.0:
        raise ValueError("--min-inlier-ratio must be between 0.0 and 1.0.")
    if args.debug_image_limit <= 0:
        raise ValueError("--debug-image-limit must be greater than zero.")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the canonical reconstruction workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_packets = iter_video_frames(
        args.video,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
    )
    options = ReconstructionOptions(
        output_dir=args.output_dir,
        min_sharpness=args.min_sharpness,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        save_debug_images=args.save_debug_images,
        debug_image_limit=args.debug_image_limit,
    )

    try:
        result = reconstruct_frames(frame_packets, options)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Video input: {args.video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Frames read: {result.frames_read}")
    print(f"Frames processed: {result.frames_processed}")
    print(f"Frames accepted: {result.frames_accepted}")
    print(f"Frames rejected: {result.frames_rejected}")
    print(f"Mean sharpness: {result.mean_sharpness:.2f}")
    print(f"Mean inlier ratio: {result.mean_inlier_ratio:.3f}")
    print(f"Reference frame saved: {result.reference_frame_saved}")
    print(f"Reference frame path: {result.reference_frame_path}")
    print(f"Final reconstruction path: {result.output_image_path}")
    print(f"Summary written: {result.summary_path}")
    print(f"Diagnostics summary written: {result.diagnostics_summary_path}")
    if result.debug_images_dir is not None:
        print(f"Debug images directory: {result.debug_images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
