#!/usr/bin/env python3
"""Eighth round: fix the failed validation, and measure the standing cost.

Two stages, both aimed at the thread axis and neither touching the clock, which
is the companion study's subject.

  duty2     The first duty-cycle run chose periods of 250 and 1000 ms for every
            model, which put four of six cells below 13% duty. There both the
            measured and the predicted energy ratio are pinned near one by the
            idle term, so the test could not discriminate: a null predictor that
            always answers 1.000 scored better than the window model on one cell
            and only 1.4 points worse overall. This picks the period per model,
            from that model's own single-thread service time, so the duty cycle
            lands near 30, 60 and 90 percent, where the window model and the
            null predictor make very different predictions.

  spinidle  The duty-cycle residuals were systematic: at four threads the board
            drew more during the idle remainder than the window model allows,
            and the excess grew as the duty cycle fell. That is what a thread
            pool still spinning after the inference returns would do. This
            measures it directly: build a session, run one inference, then
            integrate power over ten seconds of doing nothing, with the pool's
            spin-wait on and off, against a no-session control. If the pool
            spins, a duty-cycled deployment pays for it in the gaps.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, input_spec
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c

WINDOW = 30.0
TARGET_DUTY = (0.30, 0.60, 0.90)


def _session(path, threads, spin=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if spin is not None:
        so.add_session_config_entry("session.intra_op.allow_spinning", spin)
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def stage_duty2(out, reps=3):
    """Duty-cycled deployment at duty cycles where the model can be falsified."""
    import numpy as np
    seen = C.done_keys(out, ("model", "threads", "target_duty", "rep"))
    targets = ["resnet18-v1-7", "resnet50-v1-7", "googlenet-12",
               "squeezenet1.1-7"]
    set_freq(2400000)
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            for name in targets:
                path, res = C.MODELS[name]
                for duty in TARGET_DUTY:
                    for th in (1, 4):
                        key = (name, str(th), str(duty), str(rep))
                        if key in seen:
                            continue
                        cooldown(C.COOL_TO)
                        sess = _session(path, th)
                        iname, shape = input_spec(sess, res)
                        x = np.random.randn(*shape).astype(np.float32)
                        # measure this configuration's own service time, then
                        # set the period from it so the duty cycle is the
                        # controlled variable rather than the period
                        warm = []
                        t = time.perf_counter()
                        while time.perf_counter() - t < 2.0:
                            a = time.perf_counter()
                            sess.run(None, {iname: x})
                            warm.append(time.perf_counter() - a)
                        warm.sort()
                        svc = warm[len(warm) // 2]
                        period = svc / duty
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
                        rec = {"stage": "duty2", "model": name, "threads": th,
                               "target_duty": duty, "rep": rep,
                               "period_ms": 1000.0 * period,
                               "service_ms": 1000.0 * svc,
                               "window_s": t1 - t0,
                               "lat_median_ms": lat[len(lat) // 2],
                               "n_inferences": n, "n_overruns": over,
                               "duty_cycle": sum(lat) / 1000.0 / (t1 - t0),
                               "freq_khz": 2400000,
                               "freq_actual_khz": cur_freq(),
                               "temp_end_c": soc_temp_c()}
                        e = pmic.integrate(t0, t1)
                        if e:
                            rec.update(e)
                            rec["energy_per_inf_mJ"] = \
                                1000.0 * e["energy_total_J"] / n
                        del sess
                        append(out, rec)
                        print("[duty2] %-20s t=%d duty=%.2f(%.2f) r%d n=%-4d "
                              "P=%.2f W E/inf=%.1f mJ over=%d"
                              % (name, th, duty, rec["duty_cycle"], rep, n,
                                 rec.get("P_mean_W", -1),
                                 rec.get("energy_per_inf_mJ", -1), over),
                              flush=True)
            # the floor, in the same session as the runs it is subtracted from
            if ("__idle__", "0", "0", str(rep)) not in seen:
                cooldown(C.COOL_TO)
                t0 = time.perf_counter()
                time.sleep(30.0)
                t1 = time.perf_counter()
                rec = {"stage": "duty2", "model": "__idle__", "threads": 0,
                       "target_duty": 0, "rep": rep, "freq_khz": 2400000,
                       "temp_end_c": soc_temp_c()}
                e = pmic.integrate(t0, t1)
                if e:
                    rec.update(e)
                append(out, rec)
                print("[duty2] idle r%d P=%.3f W"
                      % (rep, rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


def stage_spinidle(out, reps=5):
    """What a live thread pool costs while the device is doing nothing."""
    import numpy as np
    seen = C.done_keys(out, ("model", "threads", "spin", "rep"))
    path, res = C.MODELS["resnet18-v1-7"]
    GAP = 10.0
    set_freq(2400000)
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            # control: no session at all, same board state
            if ("__none__", "0", "none", str(rep)) not in seen:
                cooldown(C.COOL_TO)
                t0 = time.perf_counter()
                time.sleep(GAP)
                t1 = time.perf_counter()
                rec = {"stage": "spinidle", "model": "__none__", "threads": 0,
                       "spin": "none", "rep": rep, "gap_s": GAP,
                       "temp_end_c": soc_temp_c()}
                e = pmic.integrate(t0, t1)
                if e:
                    rec.update(e)
                append(out, rec)
                print("[spinidle] no session      r%d P=%.4f W"
                      % (rep, rec.get("P_mean_W", -1)), flush=True)
            for th in (1, 4):
                for spin in ("1", "0"):
                    if ("resnet18-v1-7", str(th), spin, str(rep)) in seen:
                        continue
                    cooldown(C.COOL_TO)
                    sess = _session(path, th, spin)
                    iname, shape = input_spec(sess, res)
                    x = np.random.randn(*shape).astype(np.float32)
                    t = time.perf_counter()
                    while time.perf_counter() - t < 2.0:
                        sess.run(None, {iname: x})
                    # one inference, then nothing at all for GAP seconds with
                    # the session still alive
                    a = time.perf_counter()
                    sess.run(None, {iname: x})
                    b = time.perf_counter()
                    time.sleep(0.05)
                    t0 = time.perf_counter()
                    time.sleep(GAP)
                    t1 = time.perf_counter()
                    rec = {"stage": "spinidle", "model": "resnet18-v1-7",
                           "threads": th, "spin": spin, "rep": rep,
                           "gap_s": GAP, "last_inference_ms": (b - a) * 1000.0,
                           "freq_actual_khz": cur_freq(),
                           "temp_end_c": soc_temp_c()}
                    e = pmic.integrate(t0, t1)
                    if e:
                        rec.update(e)
                    del sess
                    append(out, rec)
                    print("[spinidle] t=%d spin=%s r%d P=%.4f W core=%.4f"
                          % (th, spin, rep, rec.get("P_mean_W", -1),
                             rec.get("P_core_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


STAGES = {"duty2": stage_duty2, "spinidle": stage_spinidle}

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
