# NeuralOFDM - Baseline report

This report covers the first stage of my project: a plain SISO OFDM link with QPSK,
pilots and LS channel estimation. LMMSE and the neural networks came later and are
described in their own reports.

## What I built and how to run it

The [README](README.md#project-layout) shows the project layout. The
[baseline notebook](notebooks/01_classical_ofdm_baseline.ipynb) has 14 sections and
14 code cells, and I saved its outputs after running it in a fresh kernel. All the
signal processing, simulation, experiments and plots are in `src/neural_ofdm/`, so
the notebook only calls those functions.

From the project folder:

```powershell
uv sync --locked
uv run pytest -q --basetemp=.pytest-tmp
uv run ruff check .
uv run ruff format --check .
uv run jupyter lab notebooks/01_classical_ofdm_baseline.ipynb
uv run python scripts/validate_notebook.py
uv run python scripts/run_ber_sweep.py --frames 10000 --seed 2026 --channel fixed --output results/baseline_fixed
uv run python scripts/run_ber_sweep.py --frames 10000 --seed 2026 --channel rayleigh --output results/baseline_rayleigh
```

I used Python 3.12.13 with NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2. The
exact environment is in `uv.lock` and in each sweep's `metadata.json`. I set the
pytest temporary folder inside the project because Windows was restricting the
default one. Running a fresh Jupyter kernel needs local loopback sockets to work.

## Setup and assumptions

- 64 FFT bins, a 16-sample CP and a pilot every 4 bins: 16 pilots and 48 QPSK data
  symbols, all with energy one. That gives 96 payload bits per frame.
- Every bin is used, including DC. There are no guard bands.
- No sample rate is chosen, so delays are in samples and frequencies are bin numbers.
- QPSK uses a fixed Gray mapping. The signal FFT/IFFT are unitary (`norm="ortho"`).
- The channel's frequency response is the ordinary FFT of its taps. With a CP that
  covers the channel length, this gives `Y=HX+W` in the FFT window.
- The channel convolution is a normal (linear) one, and I cut off the tail that
  goes past the frame.
- Timing is perfect and the channel does not change during a symbol.

**Channels.** The fixed channel has taps `(1, 0.35+0.15j, 0.12-0.10j)`, scaled so
their total energy is one. The Rayleigh channel has independent complex Gaussian
taps with powers `(1, 0.4, 0.15, 0.05)` (normalized). I draw a new Rayleigh channel
for every frame. The average energy is one, but I don't rescale single frames, so
some frames fade more than others. Both channels have taps at consecutive delays.

**SNR.** In this stage the SNR is the **received sample SNR**: the average clean
received power over the N+CP samples, divided by the noise variance. The noise
variance is `P_received/10^(snr_db/10)`, split equally between real and imaginary
parts. This is not Eb/N0. Because the noise is scaled to each frame's received
power, a frame with a weak channel also gets less noise. So this experiment shows
the effect of frequency selectivity, not outage at a fixed noise level. (The
classical benchmark later adds a fixed-noise mode for that.)

**LS and ZF.** LS divides each received pilot by the known pilot value. For the data
bins I interpolate the real and imaginary parts with straight lines, including the
gap that wraps around from bin 60 to bin 0. ZF divides by the channel estimate. If
an estimated gain is smaller than `1e-10` I clamp it (keeping the phase), print a
warning and count it. None of the bins needed this in either reference sweep.

**Metrics.** BER only counts payload bits. NMSE uses all 64 bins and I average it
per frame in linear units. Both receivers always see the same bits, channel and
noise. I reset the random generator to seed 2026 at each SNR, so every SNR point
uses the same random draws.

## How I checked it

The test suite had **51 passing tests** at the end of this stage, and Ruff lint and
format checks passed. I went through these checks in order and only added random
Rayleigh fading after the first ones passed:

| Check | What the test does |
|---|---|
| 1. QPSK | Four expected Gray points, unit energy, and 20,000 bits in and out without change (`test_modulation.py`) |
| 2. OFDM transforms | Pilots and data land in the right bins, FFT/IFFT round trip, energy preserved (`test_ofdm.py`) |
| 3. CP | The prefix is the right copy and is removed exactly, for CP lengths 0, 1, 16 and 64 |
| 4. Identity channel | 100 frames with no noise give BER exactly zero and the grid back within 1e-14 |
| 5. Multipath + perfect CSI | 100 noiseless frames at CP=2: BER zero, `Y` matches `HX`, equalized grid within 1e-14 |
| Two symbols in a row | With the shortest valid CP, the second symbol still satisfies `Y=HX` |
| 6. Noise + perfect CSI | Measured noise power, equal real/imaginary variance, same results for the same seed, BER falls at 0/10/20 dB |
| Theory check | Identity-channel QPSK BER matches `Q(sqrt(gamma))` within six standard errors at 0/5/10 dB |
| 7. LS receiver | Pilot division, a hand-worked interpolation example, BER=0 without noise, NMSE improving with SNR |
| Rayleigh | Measured tap powers, near-zero tap means, varying total energy, repeatable with a seed, BER=0 without noise |
| Metrics and safety | Known BER/NMSE values, bad inputs rejected, ZF clamping gives finite output and a warning |
| Saved results | The same seed gives the same results, and saved counts/settings match the run |

For the noise check I asked for 12 dB and measured **12.00633 dB** over 300,000
samples (seed 123), so the noise really has the size I think it has.

The notebook also checks things as it runs (noiseless recovery, LS values, paired
frames and the expected sweep trends). I looked over every figure for labels and
layout. Zero-error bound markers are drawn separately from the BER lines.

## Results

Seed 2026, 10,000 frames per SNR, 960,000 payload bits per SNR, SNR 0 to 30 dB in
5 dB steps, ZF for both receivers.

| Channel | SNR dB | Errors perfect / LS | BER perfect | BER LS | Mean LS NMSE |
|---|---:|---:|---:|---:|---:|
| Fixed | 0 | 167938 / 239800 | 0.17494 | 0.24979 | 0.68745 |
| Fixed | 10 | 8264 / 17683 | 0.0086083 | 0.018420 | 0.068825 |
| Fixed | 15 | 176 / 850 | 1.8333e-4 | 8.8542e-4 | 0.021825 |
| Fixed | 20 | 0 / 1 | 0 observed | 1.0417e-6 | 0.0069622 |
| Fixed | 30 | 0 / 0 | 0 observed | 0 observed | 7.7587e-4 |
| Rayleigh | 0 | 182870 / 253749 | 0.19049 | 0.26432 | 0.68418 |
| Rayleigh | 10 | 26518 / 42606 | 0.027623 | 0.044381 | 0.069473 |
| Rayleigh | 15 | 8338 / 13762 | 0.0086854 | 0.014335 | 0.022762 |
| Rayleigh | 20 | 2667 / 4661 | 0.0027781 | 0.0048552 | 0.0079868 |
| Rayleigh | 30 | 283 / 853 | 2.9479e-4 | 8.8854e-4 | 0.0018333 |

Full results and settings:

- [Fixed metrics](results/baseline_fixed/metrics.csv) and [metadata](results/baseline_fixed/metadata.json).
- [Rayleigh metrics](results/baseline_rayleigh/metrics.csv) and [metadata](results/baseline_rayleigh/metadata.json).
- [Fixed BER](results/baseline_fixed/ber_vs_snr.png), [fixed NMSE](results/baseline_fixed/nmse_vs_snr.png),
  [Rayleigh BER](results/baseline_rayleigh/ber_vs_snr.png), [Rayleigh NMSE](results/baseline_rayleigh/nmse_vs_snr.png).

BER goes down as SNR goes up on both channels, and perfect CSI is better wherever
there are enough errors to compare. LS NMSE goes down over the whole sweep. The fixed
channel is fairly mild. The Rayleigh channel has deep fades that still cause errors
at 30 dB, which makes the gap between the two receivers easier to see.

With no noise at all, LS on the fixed channel still has an NMSE of **8.8469245e-5**
(**-40.5321 dB**). At 30 dB the measured NMSE stays above this because the pilots
are still a bit noisy. This floor comes from the straight-line interpolation, not
from the CP or FFT.

Zero errors in my run doesn't mean the BER is really zero. For 960,000 bits a rough
95% upper bound is about **3.12e-6**, but that number assumes independent bits,
which isn't true with fading and channel estimation, so I only use it as a marker
on the plots. One error at 20 dB is far too little to rank the two receivers. I kept
the raw counts so this stays visible.

## Limitations

This is a small simulation: perfect timing, no Doppler, integer delays and one
symbol per channel. There is no realistic band allocation, frequency offset,
synchronization, coding or MIMO. Because the noise follows each frame's received
power, these results say nothing about outage or Eb/N0. The main things holding LS
back are the noisy pilots, the interpolation between sparse pilots, and ZF boosting
the noise in weak bins.

For the next stage I needed to pin down the channel statistics, a noise definition
that does not depend on the channel, the pilot power and the number of frames per
point, so that LMMSE could be compared fairly. That is what the
[classical benchmark](CLASSICAL_BENCHMARK_REPORT.md) does.
