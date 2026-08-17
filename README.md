# The Break-Even Parallel Speedup

**When does multithreading actually save energy on an edge CPU?**

Running inference on every core is the default on edge CPUs, on the reasoning that finishing sooner
saves energy. Sometimes it does. Sometimes it costs nearly twice as much.

Energy per inference at *n* threads relative to one obeys

```
E_n / E_1  =  (P_n / P_1) / S_n
```

so threads save energy exactly when the parallel speedup `S_n` exceeds the power ratio `P_n/P_1`.
That is an identity, not a finding. What is empirical, and what this artifact contains, is how its
two terms behave on real hardware:

- **The power ratio is nearly a board constant.** 1.59-1.99x at four threads, holding across a
  hundredfold range of work per inference.
- **The speedup is a property of the model.** 0.95x to 2.63x across architectures, and it moves
  with input resolution within a single architecture.

Because one term is a constant of the platform and the other is not, **a single calibration per
device reduces an energy question to a latency measurement** that practitioners already take.

This is the measurement artifact for a manuscript under review at *IEEE Computer Architecture
Letters*.

## What reproduces

Every headline number in the paper comes out of `data/` via `code/`. Verified on this dataset:

| Claim | Where | Value |
|---|---|---|
| Break-even threshold moves with the operating point | `analyze_freq.py` | **1.76x** at 1500 MHz to **1.86x** at 2400 MHz |
| Rule predicts the sign of the energy effect | `analyze_freq.py` | **64 of 64** model-by-thread-by-clock cases |
| Speedup range across models | `analyze_freq.py` | **0.95x to 2.63x** |
| Resolution carries one model across the threshold | `analyze_res.py` | ResNet-18 **0.72x** at 32x32 to **2.36x** at 224x224 |
| Where four threads start paying | `analyze_res.py` | 82-141 px; **four of five agree within 5 px** |
| More work alone never makes threads pay | `analyze_p2.py` (batch data) | hundredfold work increase, speedup stays ~1.10x |

Two of five architectures (ResNet-18, ResNet-50) are **still climbing at 224 px** rather than
saturating. That is in the data, and the paper says so.

## Run it

```bash
python code/analyze_freq.py data/P15_model_energy.jsonl
```

```bash
python code/analyze_res.py data/P15c_resolution.jsonl data/P15d_resolution_more.jsonl
```

Analysis needs only the Python standard library. No install step, no dependencies, no GPU.

## What is in `data/`

Every line is one measurement run, self-describing: latency median and p95, throughput, energy
integrated from the Raspberry Pi 5's on-board PMIC (total, core and DDR rails), mean power,
temperature before the run, and the exact thread count and CPU frequency it was taken at.

| File | Contents | Runs |
|---|---|---|
| `P15_model_energy.jsonl` | 8 ImageNet-scale architectures x 4 clocks x {1,2,4} threads | 288 |
| `P15e_1500mhz_recheck.jsonl` | the same 8 re-measured at 1500 MHz | 72 |
| `P15b_batch_threshold.jsonl` | MobileNetV3, batch 1 to 128 at 2400 MHz | 81 |
| `P15c_resolution.jsonl` | ResNet-18 and SqueezeNet, 32 to 224 px | 84 |
| `P15d_resolution_more.jsonl` | DenseNet, MobileNetV2, ResNet-50, 32 to 224 px | 126 |
| `P2_exit_energy.jsonl` | MobileNetV3 exit ladder at ten clocks, 1.5-2.4 GHz | 180 |

**831 runs.** The nine architectures in the paper are the eight ImageNet-scale models plus the
MobileNetV3 ladder; the four-clock sweep covers the eight.

## Hardware

Raspberry Pi 5 (Broadcom BCM2712, Cortex-A76, 4 cores), ONNX Runtime 1.24.4, models from the
[ONNX model zoo](https://github.com/onnx/models). Power is read from the board's own PMIC rather
than an external meter, so it is reproducible on any Pi 5 without extra instrumentation. CPU
frequency is pinned per run; each run reports the frequency it actually achieved alongside the one
requested.

## Scope, honestly

- One board and one microarchitecture. The **shape** of the result (power ratio nearly constant,
  speedup model-dependent) is the claim that should transfer; the specific 1.8x threshold is a
  property of this board and must be recalibrated elsewhere.
- Inference only, batch-1 latency-oriented serving unless a file says otherwise.
- The power ratio is measured at the wall of the SoC's PMIC, not per-core.

## Citation

```bibtex
@misc{jacob2026breakeven,
  author = {Jacob, Manu Nicholas},
  title  = {The Break-Even Parallel Speedup: When Multithreading Saves Energy
            for Edge CPU Inference},
  year   = {2026},
  note   = {Artifact. Manuscript under review, IEEE Computer Architecture Letters},
  url    = {https://github.com/manunicholasjacob/edge-breakeven-speedup}
}
```

ORCID [0009-0007-6589-6572](https://orcid.org/0009-0007-6589-6572)

## License

MIT, see [LICENSE](LICENSE).
