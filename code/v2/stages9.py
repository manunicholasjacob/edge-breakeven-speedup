#!/usr/bin/env python3
"""Ninth round: three controls that test the instrument rather than the model.

  purecore    The 1500 MHz block shows a second core costing about 1.9 W while
              the third costs roughly minus 0.5 W, at a firmware-confirmed
              constant 0.750 V. That is not a possible power profile for a
              homogeneous quad-core, so either the board does something we do
              not understand or the measurement does. This removes the workload
              from the question entirely: N threads spinning on a trivial
              floating-point loop, no runtime, no memory traffic, no allocator.
              If the anomaly reproduces here it is the board or the instrument;
              if it does not, it is something about the inference workload.

  samplerate  The sampler runs on the machine under test and its cost scales
              with how often it runs. If the reported mean power depends on the
              sampling period, the instrument is perturbing what it measures.
              Same cell, four sampling periods spanning an order of magnitude.

  floordrift  The idle floor moved 13% within a single session in the
              standing-cost run. The dynamic threshold subtracts the floor, so
              that drift propagates into every dynamic number in the paper.
              This characterises it: thirty-second idle windows, back to back,
              for half an hour, with temperature logged.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, input_spec
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c


def _burn(stop_at, sink):
    """A tight FP loop with no memory footprint beyond a few registers."""
    x = 1.000001
    n = 0
    while time.perf_counter() < stop_at:
        for _ in range(20000):
            x = x * 1.0000001 + 1e-9
        n += 1
    sink.append((n, x))


def stage_purecore(out, reps=5):
    import threading
    seen = C.done_keys(out, ("freq_khz", "n_threads", "rep"))
    for rep in range(1, reps + 1):
        for khz in (2400000, 1500000):
            set_freq(khz)
            for n in (0, 1, 2, 3, 4):
                if (str(khz), str(n), str(rep)) in seen:
                    continue
                cooldown(C.COOL_TO)
                pmic = PmicSampler()
                pmic.start()
                try:
                    t0 = time.perf_counter()
                    stop = t0 + 6.0
                    sink = []
                    ths = [threading.Thread(target=_burn, args=(stop, sink))
                           for _ in range(n)]
                    for th in ths:
                        th.start()
                    if n == 0:
                        time.sleep(6.0)
                    for th in ths:
                        th.join()
                    t1 = time.perf_counter()
                    rec = {"stage": "purecore", "freq_khz": khz,
                           "n_threads": n, "rep": rep,
                           "iterations": sum(s[0] for s in sink),
                           "freq_actual_khz": cur_freq(),
                           "temp_end_c": soc_temp_c()}
                    e = pmic.integrate(t0, t1)
                    if e:
                        rec.update(e)
                finally:
                    pmic.stop()
                append(out, rec)
                print("[purecore] %dMHz n=%d r%d P=%.4f W core=%.4f iters=%d"
                      % (khz // 1000, n, rep, rec.get("P_mean_W", -1),
                         rec.get("P_core_mean_W", -1), rec["iterations"]),
                      flush=True)


def stage_samplerate(out, reps=3):
    """Does the reported power depend on how often we sample it?"""
    import numpy as np
    from bench import make_session
    seen = C.done_keys(out, ("threads", "period_ms", "rep"))
    path, res = C.MODELS["resnet18-v1-7"]
    set_freq(2400000)
    for rep in range(1, reps + 1):
        for th in (1, 4):
            for per_ms in (23, 50, 100, 200):
                if (str(th), str(per_ms), str(rep)) in seen:
                    continue
                cooldown(C.COOL_TO)
                pmic = PmicSampler(period=per_ms / 1000.0)
                pmic.start()
                try:
                    sess = make_session(path, th)
                    iname, shape = input_spec(sess, res)
                    x = np.random.randn(*shape).astype(np.float32)
                    t = time.perf_counter()
                    while time.perf_counter() - t < 2.0:
                        sess.run(None, {iname: x})
                    lat = []
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < 8.0:
                        a = time.perf_counter()
                        sess.run(None, {iname: x})
                        lat.append((time.perf_counter() - a) * 1000.0)
                    t1 = time.perf_counter()
                    lat.sort()
                    rec = {"stage": "samplerate", "threads": th,
                           "period_ms": per_ms, "rep": rep,
                           "lat_median_ms": lat[len(lat) // 2],
                           "n_inferences": len(lat),
                           "temp_end_c": soc_temp_c()}
                    e = pmic.integrate(t0, t1)
                    if e:
                        rec.update(e)
                    del sess
                finally:
                    pmic.stop()
                append(out, rec)
                print("[samplerate] t=%d per=%3dms r%d P=%.4f W lat=%.2f "
                      "rate=%.1f Hz"
                      % (th, per_ms, rep, rec.get("P_mean_W", -1),
                         rec["lat_median_ms"],
                         rec.get("n_pmic_samples", 0)
                         / max(rec.get("pmic_span_s", 1), 1e-9)), flush=True)


def stage_floordrift(out, windows=40):
    """The idle floor, back to back, long enough to see it move."""
    seen = C.done_keys(out, ("idx",))
    set_freq(2400000)
    cooldown(C.COOL_TO)
    pmic = PmicSampler()
    pmic.start()
    t_start = time.perf_counter()
    try:
        for i in range(1, windows + 1):
            if (str(i),) in seen:
                continue
            t0 = time.perf_counter()
            time.sleep(30.0)
            t1 = time.perf_counter()
            rec = {"stage": "floordrift", "idx": i,
                   "elapsed_s": t1 - t_start, "freq_khz": 2400000,
                   "temp_end_c": soc_temp_c()}
            e = pmic.integrate(t0, t1)
            if e:
                rec.update(e)
            append(out, rec)
            print("[floordrift] %2d t=%6.0fs P=%.4f W core=%.4f T=%.1fC"
                  % (i, rec["elapsed_s"], rec.get("P_mean_W", -1),
                     rec.get("P_core_mean_W", -1), rec["temp_end_c"]),
                  flush=True)
    finally:
        pmic.stop()


STAGES = {"purecore": stage_purecore, "samplerate": stage_samplerate,
          "floordrift": stage_floordrift}

if __name__ == "__main__":
    import argparse
    import atexit
    from pmic import governor, limits, write_limits
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = limits()
    g0 = governor()
    atexit.register(write_limits, lo, hi)
    atexit.register(governor, g0)
    t0 = time.time()
    STAGES[a.stage](a.out)
    print("# %s complete in %.1f min" % (a.stage, (time.time() - t0) / 60.0),
          flush=True)
