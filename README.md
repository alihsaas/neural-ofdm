# NeuralOFDM

I use this project to study channel estimation and equalization for an OFDM link.

## Setup

I use Python 3.12 and uv. Install the locked dependencies with `uv sync --locked`.

## Classical simulator

I implemented a QPSK link with pilots, a cyclic prefix, multipath, and noise. Run the baseline sweep with `uv run python scripts/run_ber_sweep.py`.

## Tests

Run `uv run pytest -q --basetemp=.pytest-tmp`.

## Stage 1: baseline

```powershell
uv run jupyter lab notebooks/01_classical_ofdm_baseline.ipynb
uv run python scripts/validate_notebook.py
uv run python scripts/run_ber_sweep.py --frames 10000 --seed 2026 --channel fixed --output results/baseline_fixed
uv run python scripts/run_ber_sweep.py --frames 10000 --seed 2026 --channel rayleigh --output results/baseline_rayleigh
```

The notebook goes through the link one block at a time and saves its figures to
`results/baseline_demo/`. The sweep script also takes `--snr-start`, `--snr-stop`,
`--snr-step` and `--demo-figures` (`--help` shows the defaults). Each sweep saves:

- `metrics.csv`: bit error counts, BER, mean NMSE and how many bins ZF had to clamp;
- `metadata.json`: seed, pilot layout, channel settings, SNR definition and package versions;
- `ber_vs_snr.png` and `nmse_vs_snr.png`.

Results with seed 2026 and 10,000 frames per SNR (960,000 payload bits per point).
Both receivers see exactly the same frames and both use ZF. NMSE is linear.

| SNR (dB) | Fixed BER perfect | Fixed BER LS | Fixed LS NMSE | Rayleigh BER perfect | Rayleigh BER LS | Rayleigh LS NMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.17494 | 0.24979 | 0.68745 | 0.19049 | 0.26432 | 0.68418 |
| 10 | 0.008608 | 0.018420 | 0.068825 | 0.027623 | 0.044381 | 0.069473 |
| 15 | 0.0001833 | 0.0008854 | 0.021825 | 0.0086854 | 0.014335 | 0.022762 |
| 20 | 0 observed | 0.000001042 | 0.0069622 | 0.0027781 | 0.0048552 | 0.0079868 |
| 30 | 0 observed | 0 observed | 0.00077587 | 0.00029479 | 0.00088854 | 0.0018333 |

The CSV files have all seven SNR points. Zero errors in my run does not mean the
true BER is zero. With 960,000 bits, a rough 95% upper bound is about `3.12e-6`, and
that bound assumes independent bits, which is not really true with fading. In the
plots I show these bounds as separate triangles instead of joining them to the
BER line. The single LS error at 20 dB is too little to say anything precise.

Even with no noise, LS on the fixed channel has an NMSE of about **8.85e-5**
(-40.53 dB) because straight-line interpolation between pilots is not exact.

![Rayleigh BER comparison](results/baseline_rayleigh/ber_vs_snr.png)
