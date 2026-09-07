# NeuralOFDM

I use this project to study channel estimation and equalization for an OFDM link.

## Setup

I use Python 3.12 and uv. Install the locked dependencies with `uv sync --locked`.

## Classical simulator

I implemented a QPSK link with pilots, a cyclic prefix, multipath, and noise. Run the baseline sweep with `uv run python scripts/run_ber_sweep.py`.

## Tests

Run `uv run pytest -q --basetemp=.pytest-tmp`.
