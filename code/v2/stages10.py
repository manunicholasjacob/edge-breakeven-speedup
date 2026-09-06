#!/usr/bin/env python3
"""Tenth round: the 1500 MHz anomaly, with the workload removed.

An earlier attempt at this used Python threads, which the interpreter lock
serialises, so every thread count did the same total work on one core and the
experiment measured nothing. This uses separate processes, which the kernel is
free to place on separate cores, and verifies that assumption by reporting the
work each configuration actually completed: if N processes do not complete
roughly N times the work of one, the placement did not happen and the row is
not evidence about N active cores.

  purecore2  N processes spinning on a tight floating-point loop with no
             memory footprint, no runtime and no allocator, at both clocks.
             The thread sweep shows a second core costing about 1.9 W at
             1500 MHz while the third costs about minus 0.5 W, at a
             firmware-confirmed constant 0.750 V. Either the board does
             something we do not understand or the instrument does. This
             removes the inference workload from the question.
"""

import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c

BURN_S = 8.0


def _burn(seconds, q):
    """A tight FP loop. No allocation, no memory traffic, no syscalls."""
    x = 1.000001
    n = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(50000):
            x = x * 1.0000001 + 1e-9
        n += 1
    q.put(n)


def stage_purecore2(out, reps=5):
    seen = C.done_keys(out, ("freq_khz", "n_proc", "rep"))
    for rep in range(1, reps + 1):
        for khz in (2400000, 1500000):
            set_freq(khz)
            for n in (0, 1, 2, 3, 4):
                if (str(khz), str(n), str(rep)) in seen:
                    continue
                cooldown(C.COOL_TO)
                pmic = PmicSampler()
                pmic.start()
                rec = None
                try:
                    # let the sampler establish itself before the window opens
                    time.sleep(0.5)
                    q = mp.Queue()
                    procs = [mp.Process(target=_burn, args=(BURN_S, q))
                             for _ in range(n)]
                    t0 = time.perf_counter()
                    for p in procs:
                        p.start()
                    if n == 0:
                        time.sleep(BURN_S)
                    done = 0
                    for _ in range(n):
                        done += q.get()
                    for p in procs:
                        p.join()
                    t1 = time.perf_counter()
                    rec = {"stage": "purecore2", "freq_khz": khz,
                           "n_proc": n, "rep": rep,
                           "iterations": done,
                           "window_s": t1 - t0,
                           "freq_actual_khz": cur_freq(),
                           "temp_end_c": soc_temp_c()}
                    e = pmic.integrate(t0, t1)
                    if e:
                        rec.update(e)
                finally:
                    pmic.stop()
                if rec is None:
                    continue
                append(out, rec)
                print("[purecore2] %dMHz n=%d r%d P=%.4f core=%.4f "
                      "iters=%d nsamp=%s"
                      % (khz // 1000, n, rep, rec.get("P_mean_W", -1),
                         rec.get("P_core_mean_W", -1), rec["iterations"],
                         rec.get("n_pmic_samples", "-")), flush=True)


STAGES = {"purecore2": stage_purecore2}

if __name__ == "__main__":
    import argparse
    import atexit
    from pmic import governor, limits, write_limits
    mp.set_start_method("fork")
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
