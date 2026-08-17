#!/usr/bin/env python3
"""Locate the work-per-inference threshold at which multithreading starts to pay.

Across eight ImageNet-scale architectures four threads is a strict win: 1.7-2.6x
faster for 0.76-0.95x the energy. Across a three-exit CIFAR ladder it is a strict
loss: at most 1.05x faster for 1.55-2.00x the energy. Those two regimes sit at
0.6 ms and 14 ms of service time respectively, and the crossover is somewhere in
between.

Comparing different architectures cannot locate it, because they differ in more
than work. Instead we take a single model and sweep the batch size, which scales
work per inference continuously while holding the graph, the weights and the
memory access pattern fixed. The batch dimension is made dynamic first, so the
same file serves every batch size.

Reports per-batch latency and energy so the threshold is expressed in the unit a
deployment can actually check: milliseconds of single-threaded service time.
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


def make_batch_dynamic(src, dst):
    """Rewrite the input (and output) batch dimension to a symbolic parameter."""
    m = onnx.load(src)
    for t in list(m.graph.input) + list(m.graph.output):
        d = t.type.tensor_type.shape.dim
        if len(d) > 0:
            d[0].ClearField("dim_value")
            d[0].dim_param = "N"
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


def bench(path, batch, threads, warm_s, meas_s, pmic):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [batch] + [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape[1:]]
    x = np.random.randn(*shape).astype(np.float32)

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
    out = {"batch": batch, "n_batches": n, "wall_s": span,
           "ms_per_batch": 1000.0 * span / n,
           "ms_per_sample": 1000.0 * span / (n * batch)}
    e = pmic.integrate(t0, t1)
    if e:
        out.update(e)
        out["mJ_per_batch"] = 1000.0 * e["energy_total_J"] / n
        out["mJ_per_sample"] = 1000.0 * e["energy_total_J"] / (n * batch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--freq-khz", type=int, default=2400000)
    ap.add_argument("--warm-s", type=float, default=1.5)
    ap.add_argument("--meas-s", type=float, default=5.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dyn = os.path.join(os.path.dirname(args.out),
                       os.path.basename(args.model).replace(".onnx", "__dynbatch.onnx"))
    make_batch_dynamic(args.model, dyn)
    print(f"# batch dimension made dynamic -> {dyn}", flush=True)

    lo, hi = limits()
    atexit.register(write_limits, lo, hi)
    write_limits(args.freq_khz, args.freq_khz)
    time.sleep(0.5)

    pmic = PmicSampler()
    pmic.start()
    total = args.reps * len(args.batches) * len(args.threads)
    done = 0
    try:
        with open(args.out, "a") as fout:
            for rep in range(1, args.reps + 1):
                for b in args.batches:
                    for th in args.threads:
                        cooldown(52.0, 120.0)
                        rec = {"model": os.path.basename(args.model), "threads": th,
                               "rep": rep, "freq_khz": args.freq_khz,
                               "temp_before_c": soc_temp_c()}
                        try:
                            rec.update(bench(dyn, b, th, args.warm_s, args.meas_s, pmic))
                        except Exception as e:
                            rec.update({"batch": b, "error": str(e)[:200]})
                        fout.write(json.dumps(rec) + "\n")
                        fout.flush()
                        done += 1
                        if "error" in rec:
                            print(f"[{done}/{total}] b={b} t={th} FAILED: {rec['error'][:70]}",
                                  flush=True)
                        else:
                            print(f"[{done}/{total}] b={b:4d} t={th} r{rep}  "
                                  f"{rec['ms_per_batch']:9.3f} ms/batch  "
                                  f"{rec.get('mJ_per_batch', 0):9.3f} mJ/batch",
                                  flush=True)
    finally:
        pmic.stop()
        write_limits(lo, hi)
        print(f"# done -> {args.out}; limits restored {lo}-{hi}", flush=True)


if __name__ == "__main__":
    main()
