"""Practical smoke test for CLI orchestration path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from planar_reconstruction import cli
from planar_reconstruction.stream import FramePacket


@dataclass(frozen=True)
class _FakeResult:
    """Small stand-in for reconstruction result expected by CLI printing."""

    frames_read: int = 5
    frames_processed: int = 5
    frames_accepted: int = 4
    frames_rejected: int = 1
    mean_sharpness: float = 123.4
    mean_inlier_ratio: float = 0.75
    reference_frame_saved: bool = True
    reference_frame_path: Path | None = Path("reference_frame.png")
    output_image_path: Path | None = Path("final_reconstruction.png")
    summary_path: Path | None = Path("summary.json")
    diagnostics_summary_path: Path | None = Path("diagnostics_summary.json")
    debug_images_dir: Path | None = None


def test_cli_main_smoke_with_monkeypatched_pipeline(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """CLI should parse args, invoke pipeline path, and print summary lines."""
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"not-a-real-video")

    def fake_iter_video_frames(*_args, **_kwargs):
        return [
            FramePacket(index=0, timestamp_ms=0.0, frame=None),
            FramePacket(index=1, timestamp_ms=33.3, frame=None),
        ]

    def fake_reconstruct_frames(frame_packets, options):
        packets = list(frame_packets)
        assert len(packets) == 2
        assert options.output_dir.parent == tmp_path
        assert options.output_dir.exists()
        return _FakeResult()

    monkeypatch.setattr(cli, "iter_video_frames", fake_iter_video_frames)
    monkeypatch.setattr(cli, "reconstruct_frames", fake_reconstruct_frames)

    exit_code = cli.main(
        [
            "--video",
            str(video_path),
            "--output-dir",
            str(tmp_path),
            "--max-frames",
            "10",
            "--frame-step",
            "2",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Video input:" in out
    assert "Run output directory:" in out
    assert "Frames read:" in out
    assert "Summary written:" in out
