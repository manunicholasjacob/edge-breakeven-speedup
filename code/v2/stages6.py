#!/usr/bin/env python3
"""Sixth round: the objections that need measurement rather than rewriting.

  duty     The paper derives a window model for a duty-cycled device and then
           measures every power term back to back. This runs the deployment the
           model describes: one inference, then idle, repeated over a long
           window, with the whole window integrated. The energy ratio it gives
           can be compared against the ratio Equation (3) predicts from the
           saturated numbers, which turns the paper's thesis from an
           extrapolation into a measurement.
  clockv   `freq_actual_khz` reads back the value the harness wrote, so the
           frequency-pinning check cannot fail. This records the delivered ARM
           clock, the core voltage and the throttle flags per thread count at
           both clocks, which is also the diagnostic for the flat power block
           at 1500 MHz.
  perfN    The counter stage subtracts a startup phase whose size is
           thread-dependent and, for the CIFAR-scale network, larger than the
           residual it leaves. This scales the iteration count per target and
           takes three points instead of two, so the per-inference counts come
           from a slope rather than a difference.
  shuffle  Cell order in the main sweep is fixed, so thread count is aliased
           with position in the block and with thermal history. This repeats
           the 2400 MHz sweep in a randomised order.
"""

import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, input_spec, run_cell
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c
from stages3 import EVENTS, parse_perf

CNN9 = ["googlenet-12", "resnet18-v1-7", "efficientnet-lite4-11",
        "squeezenet1.1-7", "shufflenet-v2-10", "resnet50-v1-7",
        "mobilenetv2-12", "densenet-12", "mnv3-cifar"]


def vcgen(arg):
    try:
        out = subprocess.run(["vcgencmd"] + arg.split(), capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return out
    except Exception as exc:
        return "error: %s" % exc


# ---------------------------------------------------------------- duty ------
def stage_duty(out, reps=3):
    """One inference per period, idling in between, over a long window."""
    import numpy as np
    from bench import make_session

    seen = C.done_keys(out, ("model", "threads", "period_ms", "rep"))
    targets = ["resnet18-v1-7", "squeezenet1.1-7", "mnv3-cifar"]
    WINDOW = 30.0
    set_freq(2400000)
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            for name in targets:
                path, res = C.MODELS[name]
                for period_ms in (250, 1000):
                    for th in (1, 4):
                        key = (name, str(th), str(period_ms), str(rep))
                        if key in seen:
                            continue
                        cooldown(C.COOL_TO)
                        sess = make_session(path, th)
                        iname, shape = input_spec(sess, res)
                        x = np.random.randn(*shape).astype(np.float32)
                        t = time.perf_counter()
                        while time.perf_counter() - t < 2.0:
                            sess.run(None, {iname: x})
                        period = period_ms / 1000.0
                        lat, n, over = [], 0, 0
                        t0 = time.perf_counter()
                        nxt = t0
                        while time.perf_counter() - t0 < WINDOW:
                            nxt += period
                            a = time.perf_counter()
                            sess.run(None, {iname: x})
                            b = time.perf_counter()
                            lat.append((b - a) * 1000.0)
                            n += 1
                            if b < nxt:
                                time.sleep(nxt - b)
                            else:
                                over += 1
                        t1 = time.perf_counter()
                        lat.sort()
                        rec = {"stage": "duty", "model": name, "threads": th,
                               "period_ms": period_ms, "rep": rep,
                               "window_s": t1 - t0,
                               "lat_median_ms": lat[len(lat) // 2],
                               "n_inferences": n, "n_overruns": over,
                               "duty_cycle": sum(lat) / 1000.0 / (t1 - t0),
                               "freq_khz": 2400000,
                               "freq_actual_khz": cur_freq(),
                               "arm_clock": vcgen("measure_clock arm"),
                               "temp_start_c": None,
                               "temp_end_c": soc_temp_c()}
                        e = pmic.integrate(t0, t1)
                        if e:
                            rec.update(e)
                            rec["energy_per_inf_mJ"] = \
                                1000.0 * e["energy_total_J"] / n
                        del sess
                        append(out, rec)
                        print("[duty] %-22s t=%d T=%dms r%d n=%d duty=%.3f "
                              "P=%.2f W E/inf=%.1f mJ"
                              % (name, th, period_ms, rep, n,
                                 rec["duty_cycle"], rec.get("P_mean_W", -1),
                                 rec.get("energy_per_inf_mJ", -1)), flush=True)
            # the idle floor in the same session, so the comparison is closed
            if (("__idle__", "0", "0", str(rep))) not in seen:
                cooldown(C.COOL_TO)
                t0 = time.perf_counter()
                time.sleep(30.0)
                t1 = time.perf_counter()
                rec = {"stage": "duty", "model": "__idle__", "threads": 0,
                       "period_ms": 0, "rep": rep, "freq_khz": 2400000,
                       "temp_end_c": soc_temp_c()}
                e = pmic.integrate(t0, t1)
                if e:
                    rec.update(e)
                append(out, rec)
                print("[duty] idle r%d P=%.3f W" % (rep, rec.get("P_mean_W", -1)),
                      flush=True)
    finally:
        pmic.stop()


# -------------------------------------------------------------- clockv ------
def stage_clockv(out, reps=3):
    """What the firmware actually delivers, as opposed to what cpufreq says."""
    import numpy as np
    from bench import make_session

    seen = C.done_keys(out, ("freq_khz", "threads", "rep"))
    path, res = C.MODELS["resnet18-v1-7"]
    for rep in range(1, reps + 1):
        for khz in (2400000, 1500000):
            set_freq(khz)
            for th in (1, 2, 3, 4):
                if (str(khz), str(th), str(rep)) in seen:
                    continue
                cooldown(C.COOL_TO)
                sess = make_session(path, th)
                iname, shape = input_spec(sess, res)
                x = np.random.randn(*shape).astype(np.float32)
                t = time.perf_counter()
                while time.perf_counter() - t < 2.0:
                    sess.run(None, {iname: x})
                # sample the delivered clock and voltage under load
                samples = []
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 4.0:
                    sess.run(None, {iname: x})
                    samples.append({
                        "arm": vcgen("measure_clock arm"),
                        "core_v": vcgen("measure_volts core"),
                        "throttled": vcgen("get_throttled"),
                        "scaling_cur": cur_freq(),
                    })
                del sess
                hz = []
                for s in samples:
                    try:
                        hz.append(int(s["arm"].split("=")[1]))
                    except Exception:
                        pass
                volts = []
                for s in samples:
                    try:
                        volts.append(float(s["core_v"].split("=")[1].rstrip("V")))
                    except Exception:
                        pass
                rec = {"stage": "clockv", "freq_khz": khz, "threads": th,
                       "rep": rep, "n_samples": len(samples),
                       "arm_hz_min": min(hz) if hz else None,
                       "arm_hz_max": max(hz) if hz else None,
                       "arm_hz_med": sorted(hz)[len(hz) // 2] if hz else None,
                       "core_v_min": min(volts) if volts else None,
                       "core_v_max": max(volts) if volts else None,
                       "throttled": sorted({s["throttled"] for s in samples}),
                       "scaling_cur": sorted({s["scaling_cur"] for s in samples}),
                       "temp_end_c": soc_temp_c()}
                append(out, rec)
                print("[clockv] %dMHz t=%d r%d arm=%s..%s V=%s..%s thr=%s"
                      % (khz // 1000, th, rep, rec["arm_hz_min"],
                         rec["arm_hz_max"], rec["core_v_min"],
                         rec["core_v_max"], rec["throttled"]), flush=True)


# --------------------------------------------------------------- perfN ------
def stage_perfN(out, reps=3):
    """Counters from a slope over three iteration counts, scaled per target."""
    seen = C.done_keys(out, ("model", "res", "threads", "n_iter", "rep"))
    runner = os.path.join(os.path.expanduser("~"), "p15v2", "perf_target2.py")
    # chosen so the measured region is roughly ten seconds in every case
    targets = [("resnet18-v1-7", C.RES_MODEL, 224, (40, 80, 120)),
               ("resnet18-v1-7", C.RES_MODEL, 32, (600, 1200, 1800)),
               ("mnv3-cifar", C.MNV3, None, (1500, 3000, 4500))]
    set_freq(2400000)
    for rep in range(1, reps + 1):
        for name, path, res, iters in targets:
            for th in (1, 4):
                for n_iter in (0,) + iters:
                    if (name, str(res), str(th), str(n_iter),
                            str(rep)) in seen:
                        continue
                    cooldown(C.COOL_TO)
                    cmd = ["sudo", "perf", "stat", "-x,", "-e",
                           ",".join(EVENTS), "--", sys.executable, runner,
                           path, str(th), str(res or 0), str(n_iter)]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    body = {}
                    for line in p.stdout.splitlines():
                        if line.startswith("{"):
                            body = json.loads(line)
                    ev = parse_perf(p.stderr)
                    if not (ev and body):
                        continue
                    rec = {"stage": "perfN", "model": name, "res": res,
                           "threads": th, "n_iter": n_iter, "rep": rep,
                           "counters": {k: v[0] for k, v in ev.items()},
                           "pct_running": {k: v[1] for k, v in ev.items()},
                           "run": body, "temp_end_c": soc_temp_c()}
                    append(out, rec)
                    print("[perfN] %-14s res=%-4s t=%d n=%-5d r%d instr=%.0fM"
                          % (name, res, th, n_iter, rep,
                             ev.get("instructions", (0,))[0] / 1e6),
                          flush=True)


# ------------------------------------------------------------- shuffle ------
def stage_shuffle(out, reps=3):
    """The 2400 MHz sweep again, in a randomised order within each repetition."""
    seen = C.done_keys(out, ("model", "threads", "rep"))
    set_freq(2400000)
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            cells = [(m, th) for m in CNN9 for th in (1, 2, 3, 4)]
            random.Random(1000 + rep).shuffle(cells)
            for pos, (name, th) in enumerate(cells):
                if (name, str(th), str(rep)) in seen:
                    continue
                path, res = C.MODELS[name]
                cooldown(C.COOL_TO)
                rec = {"stage": "shuffle", "model": name, "threads": th,
                       "rep": rep, "position": pos, "freq_khz": 2400000,
                       "temp_before_c": soc_temp_c()}
                try:
                    rec.update(run_cell(path, th, pmic=pmic, res=res))
                except Exception as exc:
                    rec["error"] = str(exc)[:200]
                rec["temp_after_c"] = soc_temp_c()
                append(out, rec)
                print("[shuffle] r%d pos=%-2d %-22s t=%d %8.2f ms P=%.2f W"
                      % (rep, pos, name, th, rec.get("lat_median_ms", -1),
                         rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


STAGES = {"duty": stage_duty, "clockv": stage_clockv, "perfN": stage_perfN,
          "shuffle": stage_shuffle}

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
