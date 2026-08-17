#!/usr/bin/env python3
"""Energy-optimal early exit: is the accuracy-per-joule optimal exit the same as
the latency-optimal one?

Paper 10 established for whole-model CNN inference on this board that the
latency-optimal clock is not the energy-optimal clock, because power rises
faster than frequency while work per inference stays fixed. An early-exit ladder
adds a second axis: exit depth changes the work itself. The two axes interact,
and the question is whether the joint optimum is anywhere near either
single-axis optimum.

Sweeps (exit x CPU frequency x thread count), measuring latency and PMIC energy
per inference. Accuracy per exit is fixed and known from the ladder, so
accuracy-per-joule follows directly.

Frequency is pinned by writing scaling_max_freq and scaling_min_freq, and
restored on exit. Every cell starts from a thermal floor so DVFS headroom does
not leak between cells.
"""

import argparse
import atexit
import json
import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import PmicSampler, soc_temp_c, cooldown

CPUFREQ = "/sys/devices/system/cpu/cpu0/cpufreq"

# Measured top-1 on the full CIFAR-100 test set for the static-QDQ exits used
# here. Taken from the ladder's own characterization rather than re-measured,
# since accuracy does not depend on clock or thread count.
EXIT_ACC = {"exit1": 19.45, "exit2": 32.73, "final": 33.65}


def read_freqs():
    with open(os.path.join(CPUFREQ, "scaling_available_frequencies")) as f:
        return sorted(int(x) for x in f.read().split())


def orig_limits():
    with open(os.path.join(CPUFREQ, "scaling_min_freq")) as f:
        lo = f.read().strip()
    with open(os.path.join(CPUFREQ, "scaling_max_freq")) as f:
        hi = f.read().strip()
    return lo, hi


def set_freq(khz):
    """Pin the clock by clamping both ends of the range."""
    for cpu in range(os.cpu_count()):
        base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
        if not os.path.isdir(base):
            continue
        subprocess.run(["sudo", "tee", f"{base}/scaling_min_freq"],
                       input=str(khz).encode(), stdout=subprocess.DEVNULL, check=False)
        subprocess.run(["sudo", "tee", f"{base}/scaling_max_freq"],
                       input=str(khz).encode(), stdout=subprocess.DEVNULL, check=False)
    time.sleep(0.5)


def actual_freq_khz():
    try:
        with open(os.path.join(CPUFREQ, "scaling_cur_freq")) as f:
            return int(f.read())
    except Exception:
        return -1


def bench_exit(path, threads, n_warm, n_iter, pmic, p_idle):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    x = np.random.randn(*shape).astype(np.float32)

    for _ in range(n_warm):
        sess.run(None, {inp.name: x})

    # Measure for a fixed duration rather than a fixed iteration count. The
    # exits differ by 7x in service time and the clock sweep changes it again,
    # so a fixed count would give the fastest cells a window too short for the
    # 43 Hz PMIC to integrate meaningfully.
    t0 = time.perf_counter()
    lat = []
    n = 0
    while time.perf_counter() - t0 < n_iter:      # n_iter is seconds here
        t = time.perf_counter()
        sess.run(None, {inp.name: x})
        lat.append((time.perf_counter() - t) * 1000.0)
        n += 1
    t1 = time.perf_counter()

    e = pmic.integrate(t0, t1)
    lat.sort()
    span = t1 - t0
    out = {
        "lat_median_ms": lat[len(lat) // 2],
        "lat_mean_ms": sum(lat) / len(lat),
        "throughput_ips": n / span,
        "wall_s": span,
        "n_inferences": n,
    }
    if e:
        out.update(e)
        out["energy_per_inf_mJ"] = 1000.0 * e["energy_total_J"] / n
        dyn = max(0.0, e["energy_total_J"] - p_idle * span)
        out["dyn_energy_per_inf_mJ"] = 1000.0 * dyn / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--pattern", default="mnv3_c100_{exit}_qdq_clean.onnx")
    ap.add_argument("--exits", nargs="+", default=["exit1", "exit2", "final"])
    ap.add_argument("--freqs-khz", nargs="+", type=int, default=[])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--warm", type=int, default=150)
    ap.add_argument("--iter", type=float, default=6.0, help="measurement seconds per cell")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--p-idle", type=float, default=2.063)
    ap.add_argument("--cool-to", type=float, default=50.0)
    ap.add_argument("--cool-max", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    avail = read_freqs()
    freqs = args.freqs_khz or avail
    lo, hi = orig_limits()
    atexit.register(lambda: [
        subprocess.run(["sudo", "tee", f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_min_freq"],
                       input=lo.encode(), stdout=subprocess.DEVNULL, check=False)
        or subprocess.run(["sudo", "tee", f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_max_freq"],
                          input=hi.encode(), stdout=subprocess.DEVNULL, check=False)
        for c in range(os.cpu_count())])

    print(f"# available freqs (kHz): {avail}")
    print(f"# sweeping: {freqs}")
    print(f"# original limits restored on exit: min={lo} max={hi}")

    pmic = PmicSampler()
    pmic.start()

    total = args.reps * len(freqs) * len(args.threads) * len(args.exits)
    done = 0
    with open(args.out, "a") as fout:
        for rep in range(1, args.reps + 1):
            for khz in freqs:
                set_freq(khz)
                for th in args.threads:
                    for ex in args.exits:
                        p = os.path.join(args.models_dir, args.pattern.format(exit=ex))
                        if not os.path.exists(p):
                            print(f"  MISSING {p}")
                            continue
                        cooldown(args.cool_to, args.cool_max)
                        rec = {"exit": ex, "acc_pct": EXIT_ACC.get(ex),
                               "freq_khz": khz, "freq_actual_khz": actual_freq_khz(),
                               "threads": th, "rep": rep,
                               "temp_before_c": soc_temp_c()}
                        rec.update(bench_exit(p, th, args.warm, args.iter, pmic, args.p_idle))
                        rec["temp_after_c"] = soc_temp_c()
                        # Report accuracy per joule against TOTAL energy, not the
                        # excess over idle. At one thread the mean board power is
                        # 2.10 W against a 2.06 W idle baseline, so the dynamic
                        # component is under 2% of the signal and is dominated by
                        # drift in the baseline rather than by the workload. The
                        # dynamic figure is retained for the high-power cells but
                        # should not be used across the whole sweep.
                        if rec.get("energy_per_inf_mJ"):
                            rec["acc_per_J_total"] = rec["acc_pct"] / (rec["energy_per_inf_mJ"] / 1000.0)
                        if rec.get("dyn_energy_per_inf_mJ", 0) > 0.05 * rec.get("energy_per_inf_mJ", 1):
                            rec["acc_per_J_dyn"] = rec["acc_pct"] / (rec["dyn_energy_per_inf_mJ"] / 1000.0)
                        fout.write(json.dumps(rec) + "\n")
                        fout.flush()
                        done += 1
                        print(f"[{done}/{total}] {khz//1000:4d}MHz t={th} {ex:6s} r{rep}  "
                              f"{rec['lat_median_ms']:7.3f} ms  "
                              f"{rec.get('energy_per_inf_mJ', 0):7.3f} mJ/inf  "
                              f"P={rec.get('P_mean_W', 0):5.2f} W", flush=True)

    pmic.stop()
    print("# done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
