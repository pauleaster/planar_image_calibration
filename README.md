# Planar Image Calibration Workbench

Planar Image Calibration Workbench reconstructs a rectified 2D image of a planar
physical object from handheld video. It treats a local video file as a
deterministic frame stream and applies OpenCV-based feature detection, frame
registration, homography estimation, perspective correction, image fusion, and
validation metrics.

The project includes both:

- a CLI for repeatable deterministic runs
- a lightweight PySide6 UI for operator-driven runs

Both entry points call the same core reconstruction pipeline.

## Project Purpose

This is a compact proof-of-work for:

- planar image reconstruction from handheld video
- calibration-style diagnostics and quality checks
- practical scientific Python engineering with clear module boundaries
- a thin desktop UI wrapper over a shared pipeline

## Windows-First Setup

This project is developed and run primarily on Windows using Windows Python.

From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

Install development extras (optional, includes pytest):

```powershell
pip install -e ".[dev]"
```

## Editable Install Workflow

This repo uses a src-layout package with editable installs:

- `pip install -e .` installs runtime dependencies and links local source
- `pip install -e ".[dev]"` installs runtime + development tooling

This means changes under `src/` are immediately reflected without reinstalling.

## CLI Usage

Basic run:

```powershell
python -m planar_reconstruction.cli `
	--video data\input\20260707_132021.mp4 `
	--output-dir data\output `
	--max-frames 300 `
	--frame-step 5
```

Common tuning options:

- `--min-sharpness`
- `--min-inliers`
- `--min-inlier-ratio`
- `--save-debug-images`
- `--debug-image-limit`

## PySide6 App Usage

Launch the desktop app from PowerShell:

```powershell
python -m planar_reconstruction.app
```

In the UI:

1. Select video file
2. Select output folder
3. Click `Run`
4. Review status, final image preview, and summary metrics

The UI runs reconstruction in a background `QThread` so the window stays responsive.

## Outputs

Typical files written under your chosen output directory:

- `reference_frame.png`
- `final_reconstruction.png`
- `summary.json`
- `diagnostics_summary.json`
- `debug_images/` (only when `--save-debug-images` is enabled)

Example `summary.json` fields:

- `frames_read`
- `frames_processed`
- `frames_accepted`
- `frames_rejected`
- `mean_sharpness`
- `mean_inlier_ratio`
- `output_image`

Example `diagnostics_summary.json` additions:

- per-frame diagnostics (`index`, `timestamp_ms`, `sharpness_score`, `accepted`, `match_count`, `inlier_count`, `inlier_ratio`)
- sharpness score trace
- threshold values used for the run

## Validation Metrics and Good-Enough Criteria

The pipeline records and reports:

- frames read and processed
- frames accepted/rejected
- sharpness score distribution
- feature match count
- inlier count and inlier ratio

Current acceptance thresholds are configurable and include:

- minimum sharpness (`--min-sharpness`)
- minimum inliers (`--min-inliers`)
- minimum inlier ratio (`--min-inlier-ratio`)

Practical starting points:

- `min_sharpness`: 100.0
- `min_inliers`: 4
- `min_inlier_ratio`: 0.5

Increase thresholds when reconstructions are unstable; relax them for difficult,
low-texture or low-light captures.

## Running Tests

```powershell
python -m pytest -q
```

Current tests include:

- sharpness score behavior
- homography baseline behavior
- CLI smoke path (monkeypatched pipeline)

## Architecture Summary

Dependency direction:

- core pipeline modules in `src/planar_reconstruction/`
- CLI calls core pipeline
- UI calls core pipeline

The UI and CLI do not duplicate reconstruction logic.

## Non-Goals

This repository is intentionally not:

- a full 3D reconstruction system
- a medical imaging tool
- production-grade real-time software
- a camera SDK integration project (v1)
- a networking or distributed-processing system

WSL can be used as a convenience shell, but runtime execution is expected under
Windows Python for both CLI and UI.
