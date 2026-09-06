#!/usr/bin/env python3
"""Seventh round: enough repetitions to put intervals on the headline numbers.

The paper reports dispersion as ranges over three to six repetitions and
explicitly declines to quote confidence intervals, which is the weakest
methodological point a referee can press on. Both headline quantities are
ratios of medians, so the honest fix is more repetitions on exactly the cells
the verdict rests on, then a bootstrap.

  ci      the 2400 MHz cells the verdict depends on: all nine models at one and
          four threads, ten further repetitions each.
  idleci  ten further thirty-second idle windows at 2400 and 1500 MHz, since
          the dynamic threshold subtracts the floor and therefore inherits its
          dispersion.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, run_cell
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c

CNN9 = ["googlenet-12", "resnet18-v1-7", "efficientnet-lite4-11",
        "squeezenet1.1-7", "shufflenet-v2-10", "resnet50-v1-7",
        "mobilenetv2-12", "densenet-12", "mnv3-cifar"]


def stage_ci(out, reps=10):
    """Ten more repetitions of the one- and four-thread cells at 2400 MHz.

    Order is randomised within each repetition so thread count is not
    confounded with position, as the fixed-order sweep left it.
    """
    import random
    seen = C.done_keys(out, ("model", "threads", "rep"))
    set_freq(2400000)
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            cells = [(m, th) for m in CNN9 for th in (1, 4)]
            random.Random(7000 + rep).shuffle(cells)
            for pos, (name, th) in enumerate(cells):
                if (name, str(th), str(rep)) in seen:
                    continue
                path, res = C.MODELS[name]
                cooldown(C.COOL_TO)
                rec = {"stage": "ci", "model": name, "threads": th,
                       "rep": rep, "position": pos, "freq_khz": 2400000,
                       "temp_before_c": soc_temp_c()}
                try:
                    rec.update(run_cell(path, th, pmic=pmic, res=res))
                except Exception as exc:
                    rec["error"] = str(exc)[:200]
                rec["temp_after_c"] = soc_temp_c()
                rec["freq_actual_khz"] = cur_freq()
                append(out, rec)
                print("[ci] r%-2d pos=%-2d %-22s t=%d %8.2f ms P=%.2f W"
                      % (rep, pos, name, th, rec.get("lat_median_ms", -1),
                         rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


def stage_idleci(out, reps=10):
    """More idle windows, at both clocks the thread sweep covers."""
    seen = C.done_keys(out, ("freq_khz", "rep"))
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            for khz in (2400000, 1500000):
                if (str(khz), str(rep)) in seen:
                    continue
                set_freq(khz)
                cooldown(C.COOL_TO)
                t0 = time.perf_counter()
                time.sleep(30.0)
                t1 = time.perf_counter()
                rec = {"stage": "idleci", "freq_khz": khz, "rep": rep,
                       "freq_actual_khz": cur_freq(),
                       "temp_before_c": soc_temp_c()}
                e = pmic.integrate(t0, t1)
                if e:
                    rec.update(e)
                append(out, rec)
                print("[idleci] %dMHz r%-2d P=%.3f W core=%.3f"
                      % (khz // 1000, rep, rec.get("P_mean_W", -1),
                         rec.get("P_core_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


STAGES = {"ci": stage_ci, "idleci": stage_idleci}

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
