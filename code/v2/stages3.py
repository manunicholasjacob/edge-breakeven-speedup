#!/usr/bin/env python3
"""Third round of stages: corrected counters, and an instrument-validity check.

  perf2    Cortex-A76 counters with the counting region isolated by
           subtracting a startup-only run, and with the event set cut to what
           the PMU can count without multiplexing. The first campaign counted
           the whole process, including ONNX Runtime session construction, and
           divided by the inference count, which produced up to ten busy cores
           on a four-core part.
  isolate  The power sampler runs on the machine under test. This pins the
           sampler to one core and the workload to the others, so the reported
           power ratios can be compared against the unpinned measurement.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, input_spec, make_session
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c

# Cortex-A76 has six programmable counters plus a fixed cycle counter, so this
# set counts without multiplexing. The first campaign asked for eleven.
EVENTS = ["cycles", "instructions", "stalled-cycles-backend",
          "l1d_cache_refill", "l2d_cache_refill", "bus_access"]
N_ITER = 60


def parse_perf(stderr):
    """Return {event: (value, pct_running)} from `perf stat -x,` output."""
    out = {}
    for line in stderr.splitlines():
        f = line.split(",")
        if len(f) >= 5 and f[0] not in ("", "<not counted>", "<not supported>"):
            try:
                out[f[2]] = (float(f[0]), float(f[4]) if f[4] else 100.0)
            except (ValueError, IndexError):
                pass
    return out


def run_perf(runner, path, threads, res, n_iter):
    cmd = ["sudo", "perf", "stat", "-x,", "-e", ",".join(EVENTS), "--",
           sys.executable, runner, path, str(threads), str(res or 0),
           str(n_iter)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    body = {}
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            body = json.loads(line)
    return parse_perf(p.stderr), body


def stage_perf2(out, reps=3):
    seen = C.done_keys(out, ("model", "res", "threads", "rep"))
    runner = os.path.join(os.path.expanduser("~"), "p15v2", "perf_target2.py")
    targets = [("resnet18-v1-7", C.RES_MODEL, 224),
               ("resnet18-v1-7", C.RES_MODEL, 96),
               ("resnet18-v1-7", C.RES_MODEL, 32),
               ("mnv3-cifar", C.MNV3, None)]
    set_freq(2400000)
    for rep in range(1, reps + 1):
        for name, path, res in targets:
            for th in (1, 2, 3, 4):
                if (name, str(res), str(th), str(rep)) in seen:
                    continue
                cooldown(C.COOL_TO)
                base, _ = run_perf(runner, path, th, res, 0)
                full, body = run_perf(runner, path, th, res, N_ITER)
                if not (base and full and body):
                    continue
                delta = {k: full[k][0] - base.get(k, (0.0,))[0] for k in full}
                mux = {k: min(full[k][1], base.get(k, (100.0,))[1]) for k in full}
                cores = delta.get("cycles", 0) / (body["wall_s"] * 2.4e9)
                rec = {"stage": "perf2", "model": name, "res": res,
                       "threads": th, "rep": rep,
                       "counters": delta,
                       "counters_startup": {k: v[0] for k, v in base.items()},
                       "pct_running": mux,
                       "implied_busy_cores": cores,
                       "run": body, "freq_actual_khz": cur_freq(),
                       "temp_before_c": soc_temp_c()}
                append(out, rec)
                print("[perf2] %-14s res=%-4s t=%d r%d cores=%.2f ipc=%.2f"
                      % (name, res, th, rep, cores,
                         delta.get("instructions", 0) /
                         max(delta.get("cycles", 1), 1)), flush=True)


def stage_isolate(out, reps=3):
    """Sampler pinned to core 0, workload pinned to the rest."""
    import numpy as np
    seen = C.done_keys(out, ("model", "threads", "pinned", "rep"))
    set_freq(2400000)
    path, res = C.MODELS["resnet18-v1-7"]
    for rep in range(1, reps + 1):
        for pinned in ("1", "0"):
            for th in (1, 3):
                if ("resnet18-v1-7", str(th), pinned, str(rep)) in seen:
                    continue
                cooldown(C.COOL_TO)
                pmic = PmicSampler()
                if pinned == "1":
                    # the sampler thread pins itself; affinity is per-thread
                    orig_loop = pmic._loop

                    def pinned_loop():
                        try:
                            os.sched_setaffinity(0, {0})
                        except Exception:
                            pass
                        orig_loop()
                    pmic._loop = pinned_loop
                    os.sched_setaffinity(0, {1, 2, 3})
                else:
                    os.sched_setaffinity(0, set(range(os.cpu_count())))
                pmic.start()
                try:
                    sess = make_session(path, th)
                    name, shape = input_spec(sess, res)
                    x = np.random.randn(*shape).astype(np.float32)
                    t = time.perf_counter()
                    while time.perf_counter() - t < 2.0:
                        sess.run(None, {name: x})
                    lat, n = [], 0
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < 6.0:
                        a = time.perf_counter()
                        sess.run(None, {name: x})
                        lat.append((time.perf_counter() - a) * 1000.0)
                        n += 1
                    t1 = time.perf_counter()
                    lat.sort()
                    rec = {"stage": "isolate", "model": "resnet18-v1-7",
                           "threads": th, "pinned": pinned, "rep": rep,
                           "lat_median_ms": lat[len(lat) // 2],
                           "n_inferences": n,
                           "temp_before_c": soc_temp_c()}
                    e = pmic.integrate(t0, t1)
                    if e:
                        rec.update(e)
                    del sess
                finally:
                    pmic.stop()
                    os.sched_setaffinity(0, set(range(os.cpu_count())))
                append(out, rec)
                print("[isolate] t=%d pinned=%s r%d %8.2f ms P=%.2f W n_pmic=%d"
                      % (th, pinned, rep, rec["lat_median_ms"],
                         rec.get("P_mean_W", -1), rec.get("n_pmic_samples", -1)),
                      flush=True)


STAGES = {"perf2": stage_perf2, "isolate": stage_isolate}

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
