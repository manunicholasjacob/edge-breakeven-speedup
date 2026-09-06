#!/usr/bin/env python3
"""Fourth round: attribute the instruction growth, and bound the slow mode.

  perfspin  Counters with the runtime's spin-waiting thread pool on and off.
            perf2 showed instructions per inference growing 16% for ResNet-18
            at 224x224 but 201% for the CIFAR-scale network, with the small
            network's IPC rising and its stalls falling. That is the signature
            of a thread pool retiring cheap instructions while waiting. This
            stage tests it directly instead of inferring it.
  bimodal   The three-thread slow mode, sampled hard enough to state its
            incidence rather than describe it. Ten repetitions of one cell at
            each thread count, with no other stage interleaved.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, input_spec
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c
from stages3 import EVENTS, parse_perf

N_ITER = 60


def run_perf(runner, path, threads, res, n_iter, spin):
    cmd = ["sudo", "perf", "stat", "-x,", "-e", ",".join(EVENTS), "--",
           sys.executable, runner, path, str(threads), str(res or 0),
           str(n_iter), spin]
    p = subprocess.run(cmd, capture_output=True, text=True)
    body = {}
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            body = json.loads(line)
    return parse_perf(p.stderr), body


def stage_perfspin(out, reps=3):
    seen = C.done_keys(out, ("model", "res", "threads", "spin", "rep"))
    runner = os.path.join(os.path.expanduser("~"), "p15v2", "perf_target3.py")
    targets = [("mnv3-cifar", C.MNV3, None),
               ("resnet18-v1-7", C.RES_MODEL, 224)]
    set_freq(2400000)
    for rep in range(1, reps + 1):
        for name, path, res in targets:
            for th in (1, 4):
                for spin in ("1", "0"):
                    if (name, str(res), str(th), spin, str(rep)) in seen:
                        continue
                    cooldown(C.COOL_TO)
                    base, _ = run_perf(runner, path, th, res, 0, spin)
                    full, body = run_perf(runner, path, th, res, N_ITER, spin)
                    if not (base and full and body):
                        continue
                    delta = {k: full[k][0] - base.get(k, (0.0,))[0]
                             for k in full}
                    mux = {k: min(full[k][1], base.get(k, (100.0,))[1])
                           for k in full}
                    cores = delta.get("cycles", 0) / (body["wall_s"] * 2.4e9)
                    rec = {"stage": "perfspin", "model": name, "res": res,
                           "threads": th, "spin": spin, "rep": rep,
                           "counters": delta,
                           "counters_startup": {k: v[0]
                                                for k, v in base.items()},
                           "pct_running": mux,
                           "implied_busy_cores": cores,
                           "run": body, "freq_actual_khz": cur_freq(),
                           "temp_before_c": soc_temp_c()}
                    append(out, rec)
                    print("[perfspin] %-13s t=%d spin=%s r%d %8.2f ms "
                          "instr/inf=%.1fM ipc=%.2f"
                          % (name, th, spin, rep, body["ms_per_inf"],
                             delta.get("instructions", 0) / N_ITER / 1e6,
                             delta.get("instructions", 0) /
                             max(delta.get("cycles", 1), 1)), flush=True)


def stage_bimodal(out, reps=10):
    """Incidence of the slow mode, at one cell per thread count."""
    import numpy as np
    from bench import make_session
    seen = C.done_keys(out, ("model", "threads", "rep"))
    set_freq(2400000)
    path, res = C.MODELS["resnet18-v1-7"]
    for rep in range(1, reps + 1):
        for th in (2, 3, 4):
            if ("resnet18-v1-7", str(th), str(rep)) in seen:
                continue
            cooldown(C.COOL_TO)
            pmic = PmicSampler()
            pmic.start()
            try:
                sess = make_session(path, th)
                name, shape = input_spec(sess, res)
                x = np.random.randn(*shape).astype(np.float32)
                t = time.perf_counter()
                while time.perf_counter() - t < 2.0:
                    sess.run(None, {name: x})
                lat = []
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 6.0:
                    a = time.perf_counter()
                    sess.run(None, {name: x})
                    lat.append((time.perf_counter() - a) * 1000.0)
                t1 = time.perf_counter()
                lat.sort()
                rec = {"stage": "bimodal", "model": "resnet18-v1-7",
                       "threads": th, "rep": rep,
                       "lat_median_ms": lat[len(lat) // 2],
                       "lat_p95_ms": lat[int(0.95 * len(lat))],
                       "n_inferences": len(lat),
                       "freq_actual_khz": cur_freq(),
                       "temp_before_c": soc_temp_c()}
                e = pmic.integrate(t0, t1)
                if e:
                    rec.update(e)
                del sess
            finally:
                pmic.stop()
            append(out, rec)
            print("[bimodal] t=%d r%-2d %8.2f ms P=%.2f W T=%.1fC"
                  % (th, rep, rec["lat_median_ms"], rec.get("P_mean_W", -1),
                     rec.get("temp_before_c", -1)), flush=True)


STAGES = {"perfspin": stage_perfspin, "bimodal": stage_bimodal}

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
