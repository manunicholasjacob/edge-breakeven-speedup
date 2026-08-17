#!/usr/bin/env python3
"""Isolate what actually sets the parallel speedup: input spatial resolution.

Two workload groups landed on opposite sides of the energy break-even, and a
batch sweep ruled out work volume as the explanation. The remaining hypothesis is
that intra-operator parallelism follows the spatial extent of each convolution,
so a 224x224 network parallelises and a 32x32 one does not, whatever the total
work.

That hypothesis is testable directly: take ONE architecture, make its input
spatial dimensions dynamic, and sweep resolution while holding weights, depth,
operator types and thread count fixed. If speedup tracks resolution, the
mechanism is settled. If it does not, the hypothesis is wrong and the paper says
so.
"""

import argparse
import atexit
import json
import os
import subprocess
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import PmicSampler, soc_temp_c, cooldown

ort.set_default_logger_severity(3)
CPUFREQ = "/sys/devices/system/cpu/cpu0/cpufreq"


def make_spatial_dynamic(src, dst):
    """Make N, H and W symbolic on the input so one file serves every resolution."""
    m = onnx.load(src)
    t = m.graph.input[0]
    d = t.type.tensor_type.shape.dim
    for i, name in ((0, "N"), (2, "H"), (3, "W")):
        if len(d) > i:
            d[i].ClearField("dim_value")
            d[i].dim_param = name
    # outputs must be free to follow
    for o in m.graph.output:
        od = o.type.tensor_type.shape.dim
        if len(od) > 0:
            od[0].ClearField("dim_value")
            od[0].dim_param = "N"
    onnx.save(m, dst)
    return dst


def limits():
    with open(os.path.join(CPUFREQ, "scaling_min_freq")) as f:
        lo = f.read().strip()
    with open(os.path.join(CPUFREQ, "scaling_max_freq")) as f:
        hi = f.read().strip()
    return lo, hi


def write_limits(lo, hi):
    for cpu in range(os.cpu_count()):
        b = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq"
        if not os.path.isdir(b):
            continue
        for n, v in (("scaling_min_freq", lo), ("scaling_max_freq", hi)):
            subprocess.run(["sudo", "tee", f"{b}/{n}"], input=str(v).encode(),
                           stdout=subprocess.DEVNULL, check=False)


def bench(path, res, threads, warm_s, meas_s, pmic):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.random.randn(1, 3, res, res).astype(np.float32)

    t = time.perf_counter()
    while time.perf_counter() - t < warm_s:
        sess.run(None, {inp.name: x})

    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < meas_s:
        sess.run(None, {inp.name: x})
        n += 1
    t1 = time.perf_counter()

    span = t1 - t0
    out = {"resolution": res, "n_inferences": n, "ms_per_inf": 1000.0 * span / n,
           "wall_s": span}
    e = pmic.integrate(t0, t1)
    if e:
        out.update(e)
        out["mJ_per_inf"] = 1000.0 * e["energy_total_J"] / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--resolutions", nargs="+", type=int,
                    default=[32, 48, 64, 96, 128, 160, 224])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--freq-khz", type=int, default=2400000)
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--meas-s", type=float, default=5.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dyn = {}
    for m in args.models:
        d = os.path.join(os.path.dirname(args.out),
                         os.path.basename(m).replace(".onnx", "__dynspatial.onnx"))
        try:
            dyn[m] = make_spatial_dynamic(m, d)
            print(f"# {os.path.basename(m)} -> dynamic H,W", flush=True)
        except Exception as e:
            print(f"# {os.path.basename(m)} SKIPPED: {str(e)[:80]}", flush=True)

    lo, hi = limits()
    atexit.register(write_limits, lo, hi)
    write_limits(args.freq_khz, args.freq_khz)
    time.sleep(0.5)

    pmic = PmicSampler()
    pmic.start()
    total = args.reps * len(args.resolutions) * len(args.threads) * len(dyn)
    done = 0
    try:
        with open(args.out, "a") as fout:
            for rep in range(1, args.reps + 1):
                for m, path in dyn.items():
                    for res in args.resolutions:
                        for th in args.threads:
                            cooldown(52.0, 120.0)
                            rec = {"model": os.path.basename(m), "threads": th,
                                   "rep": rep, "freq_khz": args.freq_khz,
                                   "temp_before_c": soc_temp_c()}
                            try:
                                rec.update(bench(path, res, th, args.warm_s,
                                                 args.meas_s, pmic))
                            except Exception as e:
                                rec.update({"resolution": res, "error": str(e)[:160]})
                            fout.write(json.dumps(rec) + "\n")
                            fout.flush()
                            done += 1
                            if "error" in rec:
                                print(f"[{done}/{total}] {rec['model'][:18]:18s} r={res} t={th} "
                                      f"FAILED", flush=True)
                            else:
                                print(f"[{done}/{total}] {rec['model'][:18]:18s} r={res:3d} "
                                      f"t={th} {rec['ms_per_inf']:8.2f} ms "
                                      f"{rec.get('mJ_per_inf', 0):8.2f} mJ", flush=True)
    finally:
        pmic.stop()
        write_limits(lo, hi)
        print(f"# done -> {args.out}; limits restored {lo}-{hi}", flush=True)


if __name__ == "__main__":
    main()
