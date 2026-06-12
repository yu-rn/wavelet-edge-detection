# Project2: Wavelet-Based Edge Detection

## Overview

This project is a computer vision course Project2 on edge detection based on wavelet transform.

Current dataset layout keeps the original files unchanged:

- Input images: `samples/images/`
- Ground Truth: `samples/groundTruth/`

All generated outputs are saved under `results/`.

## Environment

Recommended Python version:

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Directory Layout

```text
project_root/
  samples/
    images/
    groundTruth/
  src/
    wavelet_edge.py
    evaluate.py
    visualize.py
    run_all.py
  results/
    edges/
    intermediate/
    comparisons/
    metrics/
  README.md
  requirements.txt
  AGENTS.md
```

## Run

Run the full pipeline from the project root:

```bash
python -m src.run_all
```

Optional arguments:

```bash
python -m src.run_all --image-dir samples/images --gt-dir samples/groundTruth --output-dir results
```

## Outputs

All outputs are written to `results/`:

- `results/edges/`: final binary edge maps
- `results/intermediate/`: grayscale, wavelet detail response, normalized maps
- `results/comparisons/`: side-by-side comparison figures
- `results/metrics/`: per-image and summary metrics tables

## Ground Truth Matching

The current code assumes the following naming rule:

- image: `106025.jpg`
- GT binary: `106025_gt_binary.png`

The helper logic uses the image stem and appends `_gt_binary.png` for evaluation.

## Notes

- Original sample and Ground Truth files are not moved or modified.
- All paths are handled relative to the project root.
- The current implementation is a clean baseline scaffold for the course project and can be extended incrementally.
