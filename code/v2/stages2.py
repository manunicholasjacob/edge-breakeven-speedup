#!/usr/bin/env python3
"""Second round of stages, added after review of the first campaign.

Each answers a specific defect the reviewers found:

  cotenant2  the first co-tenancy run let the load generator run for 30 s
             against a 6 s measurement window and compared it to an 8 s
             reference, so the throughput share was an artifact of two
             hardcoded durations. Here every condition runs for the same
             window and the co-tenant is measured over exactly that window.
  ops3       the per-operator profiles were single runs.
  sustained3 the five-minute runs were single runs.
  recheck15  at 1500 MHz the two-thread board power exceeded the three-thread
             power, which is not physical. Re-measured on a quiet board.
"""

import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, input_spec, make_session, run_cell
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c

WINDOW = 12.0        # one window for every co-tenancy condition
COOL_TO = C.COOL_TO


def bogo(out_text):
    m = re.search(r"cpu\s+(\d+)\s", out_text or "")
    return int(m.group(1)) if m else None


def stage_cotenant2(out, reps=3):
    """All four conditions share one measurement window of WINDOW seconds."""
    seen = C.done_keys(out, ("cond", "rep"))
    path, res = C.MODELS["resnet18-v1-7"]
    pmic = PmicSampler()
    pmic.start()
    set_freq(2400000)
    try:
        for rep in range(1, reps + 1):
            for cond in ("infer4_alone", "infer1_alone",
                         "infer1_plus_3cotenant", "cotenant3_alone",
                         "cotenant4_alone"):
                if (cond, str(rep)) in seen:
                    continue
                cooldown(COOL_TO)
                rec = {"stage": "cotenant2", "cond": cond, "rep": rep,
                       "window_s": WINDOW, "temp_before_c": soc_temp_c()}
                hog = None
                n_hog = 0
                if cond.startswith("cotenant"):
                    n_hog = int(cond[8])
                elif cond == "infer1_plus_3cotenant":
                    n_hog = 3
                if n_hog:
                    # the co-tenant runs for exactly the window, started just
                    # before it and harvested just after
                    hog = subprocess.Popen(
                        ["stress-ng", "--cpu", str(n_hog),
                         "--timeout", "%ds" % int(WINDOW),
                         "--metrics-brief"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True)
                    time.sleep(0.3)
                if cond.startswith("cotenant"):
                    t0 = time.perf_counter()
                    time.sleep(WINDOW)
                    t1 = time.perf_counter()
                    e = pmic.integrate(t0, t1)
                    if e:
                        rec.update(e)
                else:
                    th = 1 if cond.startswith("infer1") else 4
                    try:
                        rec.update(run_cell(path, th, meas_s=WINDOW,
                                            warm_s=2.0, pmic=pmic, res=res))
                    except Exception as exc:
                        rec["error"] = str(exc)[:200]
                if hog is not None:
                    try:
                        o = hog.communicate(timeout=WINDOW + 30)[0]
                        rec["cotenant_bogo_ops"] = bogo(o)
                        rec["cotenant_cores"] = n_hog
                        b = rec["cotenant_bogo_ops"]
                        if b:
                            rec["cotenant_bogo_per_core_s"] = b / WINDOW / n_hog
                    except Exception:
                        hog.kill()
                rec["temp_after_c"] = soc_temp_c()
                append(out, rec)
                print("[cotenant2] %-22s r%d lat=%s P=%.2f bogo/core/s=%s"
                      % (cond, rep, rec.get("lat_median_ms", "-"),
                         rec.get("P_mean_W", -1),
                         rec.get("cotenant_bogo_per_core_s")), flush=True)
    finally:
        pmic.stop()


def stage_ops3(out, reps=3):
    import numpy as np
    seen = C.done_keys(out, ("res", "threads", "rep"))
    set_freq(2400000)
    for rep in range(1, reps + 1):
        for res in (32, 96, 224):
            for th in (1, 4):
                if (str(res), str(th), str(rep)) in seen:
                    continue
                cooldown(COOL_TO)
                sess = make_session(C.RES_MODEL, th, profile=True)
                name, shape = input_spec(sess, res)
                x = np.random.randn(*shape).astype(np.float32)
                for _ in range(30):
                    sess.run(None, {name: x})
                prof = sess.end_profiling()
                with open(prof) as f:
                    events = json.load(f)
                agg = {}
                for e in events:
                    if e.get("cat") == "Node" and "dur" in e:
                        op = e.get("args", {}).get("op_name", "?")
                        a = agg.setdefault(op, {"n": 0, "us": 0.0})
                        a["n"] += 1
                        a["us"] += e["dur"]
                os.remove(prof)
                append(out, {"stage": "ops3", "model": "resnet18-v1-7",
                             "res": res, "threads": th, "rep": rep,
                             "shape": shape, "ops": agg})
                print("[ops3] res=%d t=%d r%d node=%.1f ms"
                      % (res, th, rep, sum(v["us"] for v in agg.values()) / 1000),
                      flush=True)
                del sess


def stage_sustained3(out, reps=3, seconds=180.0):
    seen = C.done_keys(out, ("model", "threads", "rep"))
    pmic = PmicSampler()
    pmic.start()
    set_freq(2400000)
    try:
        for rep in range(1, reps + 1):
            for name, path, res in (("resnet18-v1-7", C.RES_MODEL, 224),
                                    ("mnv3-cifar", C.MNV3, None)):
                for th in (1, 4):
                    if (name, str(th), str(rep)) in seen:
                        continue
                    cooldown(COOL_TO)
                    rec = {"stage": "sustained3", "model": name, "threads": th,
                           "rep": rep, "seconds": seconds,
                           "temp_before_c": soc_temp_c()}
                    try:
                        rec.update(run_cell(path, th, meas_s=seconds,
                                            warm_s=3.0, pmic=pmic, res=res))
                    except Exception as e:
                        rec["error"] = str(e)[:200]
                    rec["temp_after_c"] = soc_temp_c()
                    append(out, rec)
                    print("[sustained3] %-16s t=%d r%d %.2f ms P=%.2f W"
                          % (name, th, rep, rec.get("lat_median_ms", -1),
                             rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


def stage_recheck15(out, reps=5):
    """Is the 1500 MHz two-thread power really above the three-thread power?"""
    seen = C.done_keys(out, ("model", "threads", "rep"))
    pmic = PmicSampler()
    pmic.start()
    set_freq(1500000)
    try:
        for rep in range(1, reps + 1):
            for th in (1, 2, 3, 4):
                for name in ("resnet18-v1-7", "googlenet-12", "squeezenet1.1-7"):
                    if (name, str(th), str(rep)) in seen:
                        continue
                    path, res = C.MODELS[name]
                    cooldown(COOL_TO)
                    rec = {"stage": "recheck15", "model": name,
                           "freq_khz": 1500000, "rep": rep,
                           "temp_before_c": soc_temp_c()}
                    try:
                        rec.update(run_cell(path, th, pmic=pmic, res=res))
                    except Exception as e:
                        rec["error"] = str(e)[:200]
                    rec["temp_after_c"] = soc_temp_c()
                    append(out, rec)
                    print("[recheck15] t=%d %-20s r%d %8.2f ms P=%.2f W"
                          % (th, name, rep, rec.get("lat_median_ms", -1),
                             rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


STAGES = {"cotenant2": stage_cotenant2, "ops3": stage_ops3,
          "sustained3": stage_sustained3, "recheck15": stage_recheck15}

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
