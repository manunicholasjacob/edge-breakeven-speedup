# The Break-Even Parallel Speedup

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21987261.svg)](https://doi.org/10.5281/zenodo.21987261)

**When does multithreading actually save energy on an edge CPU?**

Running inference on every core is the default on edge CPUs, on the reasoning that finishing sooner
saves energy. Whether it does turns on one comparison, and on an accounting choice that is usually
left implicit.

Energy per inference at *n* threads relative to one obeys

```
E_n / E_1  =  (P_n / P_1) / S_n
```

so threads save energy exactly when the parallel speedup `S_n` exceeds the power ratio `P_n/P_1`.
That is an identity, not a finding. What is empirical is how its two terms behave on real hardware,
and what is counted in `P`.

## Two campaigns, and a reversal

This repository holds **two** campaigns on the same Raspberry Pi 5. They reach opposite
conclusions, and the second corrects the first.

### `data/` and `code/` — the first campaign (superseded)

Six files, 831 PMIC-measured runs. It charged the board's static power to the inference, on the
stated grounds that the dynamic term was "under 2% of the signal", and concluded that eight
ImageNet-scale networks clear the break-even bar and save 5-24% energy at four threads.

**That justification was wrong.** The 2% figure came from the dataset's *minimum* cell. Measuring
the static floor at each clock instead of assuming it gives 1.72-2.00 W, so the dynamic term is
44-69% of the signal, not 2%.

### `data/v2/` and `code/v2/` — the second campaign (current)

Twenty-six files from a much larger campaign, and the basis of the current manuscript. It measures
the static floor at every operating point rather than assuming it, which yields *two* break-even
thresholds rather than one:

- **Charging the floor** (correct only for a device that never idles): break-even is 1.82x, every
  ImageNet-scale network clears it, saving 12-27% of board energy.
- **Excluding it** (what a window model requires whenever the device has idle slack): each network's
  bar sits above its own speedup. At four threads **none of the eight clears**, and bootstrap
  intervals over ten repetitions per cell put the *entire* 95% interval above break-even for all
  eight.

The floor at which each network's decision would flip is 0.82-1.46 W, against a measured floor of
1.88-2.03 W, so the verdict does not depend on knowing the floor precisely.

**Mechanism.** Instructions retired per inference grow with thread count on a fixed graph.
Disabling the runtime's spin-waiting thread pool removes 94% of that growth while making inference
slower: the runtime buys latency with energy. Cache contention is a separable second mechanism that
survives the switch. Per-operator profiles and an input-resolution sweep locate the available
parallelism in operator spatial extent rather than model size or total work.

## What is in `data/v2/`

Every repetition ships, including the ones the manuscript identifies as contaminated, and including
the stages whose results were negative or which refuted the hypothesis they were built to test.

| file | stage |
|---|---|
| `E_idle`, `E_idlegov`, `E_idleE`, `E_idleci` | static floor, at four clocks, two governors and in each measurement session |
| `E_threads`, `E_threads3`, `E_recheck15` | the 1-4 thread sweep at two clocks, plus re-measurements |
| `E_shuffle` | the same sweep in randomised cell order |
| `E_ci` | ten repetitions of every 1- and 4-thread cell, for bootstrap intervals |
| `E_perf`, `E_perf2`, `E_perfN`, `E_perfspin` | Arm PMU counters: whole-process (superseded), startup-subtracted, iteration-count regression, and with the spin-wait disabled |
| `E_spin`, `E_spinE`, `E_spinidle` | the runtime's spin-wait: latency, energy, and standing cost while idle |
| `E_ops`, `E_ops3` | per-operator profiles at three input resolutions |
| `E_gemm`, `E_llm` | a dense BLAS GEMM and llama.cpp decode |
| `E_cotenant`, `E_cotenant2` | co-tenancy, with the second run correcting a window mismatch in the first |
| `E_sustained`, `E_sustained3`, `E_governor` | five-minute runs and governor interaction |
| `E_duty`, `E_duty2`, `E_duty3` | duty-cycled deployment; see the note below |
| `E_clockv` | delivered ARM clock and core voltage read from the firmware |
| `E_bimodal`, `E_isolate` | a slow-mode incidence run, and the sampler pinned off the workload's cores |

**A note on `E_duty2`.** That run holds the *duty cycle* constant rather than the period, which
means the two thread counts do not serve the same number of inferences and their energy-per-inference
ratio is not comparable. It is kept because it was run, and because the design error is instructive.
`E_duty3` is the corrected version: the period is fixed from the one-thread service time and used
for both arms.

## Reproducing

`code/v2/fill_numbers_v2.py` regenerates every number in the manuscript from `data/v2/` into a
single LaTeX macro file, so no figure or table is hand transcribed. `code/v2/figs_v2.py` regenerates
the figures. The acquisition harnesses are `campaign.py` and `stages2.py` through `stages11.py`;
`bench.py` and `pmic.py` are the shared benchmark core and power sampler.

## Platform

Raspberry Pi 5 (Broadcom BCM2712, quad-core Arm Cortex-A76, 2 GB LPDDR4X), 64-bit Raspberry Pi OS,
ONNX Runtime 1.24.3. Power is the sum of the on-board PMIC's monitored rails, read through
`vcgencmd pmic_read_adc` at about 21 Hz and integrated trapezoidally. That instrument is downstream
of the board's converters and runs on the machine under test; the manuscript's limitations section
says what follows from both.

## License

MIT. See `LICENSE`.
