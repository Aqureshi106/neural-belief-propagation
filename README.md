# CSCI 381 Final Project: Neural Belief Propagation

This repository contains my CSCI 381 final project on neural decoding for the BCH(15,11) code over a binary-input AWGN channel.

## What is included

- `week1.py`: BCH encoder, Tanner graph construction, BI-AWGN channel, classical decoders, and ML lower bound
- `week2_stage_one.py`: PyTorch port of Sum-Product and Min-Sum belief propagation
- `week2_stage_two.py`: trainable Neural BP model and training loop
- `week2_corrected_eval.py`: fair T=5 evaluation with shared Monte Carlo samples and Wilson confidence intervals
- `week2_diagnostics.py`: diagnostic checks for the Stage 2 model
- `requirements.txt`: Python dependencies

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## How to run

Run the Week 1 baseline checks:

```bash
python week1_bch_baselines.py
```

Run the Stage 1 PyTorch equivalence tests:

```bash
python week2_stage1_pytorch_port.py
```

Run the Stage 2 neural decoder and final evaluation:

```bash
python week2_corrected_eval.py
```

## Output figures

The final evaluation script regenerates:

- `bler_vs_snr.png`
- `learned_weights_hist.png`

## Notes

- The repository is configured to ignore local virtual environments, caches, and other generated files.
- The project uses a fixed random seed for reproducibility.