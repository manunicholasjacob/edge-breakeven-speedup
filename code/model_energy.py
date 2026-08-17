#!/usr/bin/env python3
"""Energy-optimal configuration for CNN inference on an edge CPU.

Generalises the early-exit energy sweep to arbitrary models: for each
(model x clock x thread count) it measures latency and total board energy per
inference, so the joint optimum can be located rather than assumed.

Total board energy is the reported metric. The excess over idle is not usable at
low thread counts on this board: mean power there is about 2.1 W against a 2.06 W
idle baseline, so the dynamic term is a few percent of the signal and is
dominated by baseline drift. Total energy is also what a battery pays.

Frequency is pinned by clamping both ends of the scaling range and restored on
exit, including on exception, since leaving a board clamped would silently
corrupt every later measurement.
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

ort.set_default_logger_severity(3)
CPUFREQ = "/sys/devices/system/cpu/cpu0/cpufreq"


def read_freqs():
    with open(os.path.join(CPUFREQ, "scaling_available_frequencies")) as f:
        return sorted(int(x) for x in f.read().split())


def limits():
    with open(os.path.join(CPUFREQ, "scaling_min_freq")) as f:
        lo = f.read().strip()
    with open(os.path.join(CPUFREQ, "scaling_max_freq")) as f:
        hi = f.read().strip()
    return lo, hi


def write_limits(lo, hi):
    for cpu in range(os.cpu_count()):
        base = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
        if not os.path.isdir(base):
            continue
        for name, val in (("scaling_min_freq", lo), ("scaling_max_freq", hi)):
            subprocess.run(["sudo", "tee", f"{base}/{name}"], input=str(val).encode(),
                           stdout=subprocess.DEVNULL, check=False)


def set_freq(khz):
    write_limits(khz, khz)
    time.sleep(0.4)


def cur_freq():
    try:
        with open(os.path.join(CPUFREQ, "scaling_cur_freq")) as f:
            return int(f.read())
    except Exception:
        return -1


def bench(path, threads, warm_s, meas_s, pmic, p_idle):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    x = np.random.randn(*shape).astype(np.float32)

    t = time.perf_counter()
    while time.perf_counter() - t < warm_s:
        sess.run(None, {inp.name: x})

    t0 = time.perf_counter()
    lat, n = [], 0
    while time.perf_counter() - t0 < meas_s:
        a = time.perf_counter()
        sess.run(None, {inp.name: x})
        lat.append((time.perf_counter() - a) * 1000.0)
        n += 1
    t1 = time.perf_counter()

    lat.sort()
    span = t1 - t0
    out = {"lat_median_ms": lat[len(lat) // 2], "lat_p95_ms": lat[int(0.95 * len(lat))],
           "n_inferences": n, "wall_s": span, "throughput_ips": n / span,
           "input_shape": shape, "file_mb": os.path.getsize(path) / 1048576.0}
    e = pmic.integrate(t0, t1)
    if e:
        out.update(e)
        out["energy_per_inf_mJ"] = 1000.0 * e["energy_total_J"] / n
        dyn = max(0.0, e["energy_total_J"] - p_idle * span)
        out["dyn_energy_per_inf_mJ"] = 1000.0 * dyn / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--freqs-khz", nargs="+", type=int, default=[])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--warm-s", type=float, default=2.0)
    ap.add_argument("--meas-s", type=float, default=6.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--p-idle", type=float, default=2.063)
    ap.add_argument("--cool-to", type=float, default=52.0)
    ap.add_argument("--cool-max", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    avail = read_freqs()
    freqs = args.freqs_khz or avail
    lo, hi = limits()
    atexit.register(write_limits, lo, hi)
    print(f"# available {avail}\n# sweeping {freqs}\n# will restore {lo}-{hi} on exit", flush=True)

    pmic = PmicSampler()
    pmic.start()
    total = args.reps * len(freqs) * len(args.threads) * len(args.models)
    done = 0
    try:
        with open(args.out, "a") as fout:
            for rep in range(1, args.reps + 1):
                for khz in freqs:
                    set_freq(khz)
                    for th in args.threads:
                        for m in args.models:
                            if not os.path.exists(m):
                                continue
                            cooldown(args.cool_to, args.cool_max)
                            rec = {"model": os.path.splitext(os.path.basename(m))[0],
                                   "freq_khz": khz, "freq_actual_khz": cur_freq(),
                                   "threads": th, "rep": rep,
                                   "temp_before_c": soc_temp_c()}
                            try:
                                rec.update(bench(m, th, args.warm_s, args.meas_s, pmic, args.p_idle))
                            except Exception as e:
                                rec["error"] = str(e)[:200]
                            rec["temp_after_c"] = soc_temp_c()
                            fout.write(json.dumps(rec) + "\n")
                            fout.flush()
                            done += 1
                            if "error" in rec:
                                print(f"[{done}/{total}] {rec['model']:22s} FAILED", flush=True)
                            else:
                                print(f"[{done}/{total}] {khz//1000:4d}MHz t={th} "
                                      f"{rec['model']:22s} r{rep} "
                                      f"{rec['lat_median_ms']:8.3f} ms "
                                      f"{rec.get('energy_per_inf_mJ', 0):8.3f} mJ "
                                      f"P={rec.get('P_mean_W', 0):5.2f} W", flush=True)
    finally:
        pmic.stop()
        write_limits(lo, hi)
        print(f"# done -> {args.out}; frequency limits restored to {lo}-{hi}", flush=True)


if __name__ == "__main__":
    main()
