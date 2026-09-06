#!/usr/bin/env python3
"""Eleventh round: the duty-cycle test, with the comparison made correctly.

Two earlier attempts at this got the design wrong in opposite ways.

The first fixed the period at 250 and 1000 ms for every model, which is the
right comparison (a fixed period means both thread counts serve the same number
of inferences in the same window) but put four of six cells below 13% duty,
where the idle term dominates and both the measured and predicted ratios are
pinned near one. Only the single high-duty cell discriminated, and it agreed to
0.2%.

The second fixed the duty cycle instead, deriving each configuration's period
from its own service time. That is not the same comparison: holding duty
constant while the service time falls shrinks the period, so the four-thread
arm serves roughly two and a half times as many inferences as the one-thread
arm. Energy per inference then falls for arithmetic reasons that have nothing
to do with the model, which is what the 0.54 ratios in that run are.

This does it properly: the period is computed once, from the ONE-THREAD service
time, and the SAME period is used for both thread counts, so W and T are
identical across the comparison. Periods are chosen to put the one-thread duty
cycle at 0.9, 0.6 and 0.3, which is where the window model and a null predictor
disagree.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, input_spec
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c

WINDOW = 30.0
ONE_THREAD_DUTY = (0.90, 0.60, 0.30)


def stage_duty3(out, reps=3):
    import numpy as np
    import onnxruntime as ort

    def _sess(path, threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(path, sess_options=so,
                                    providers=["CPUExecutionProvider"])

    seen = C.done_keys(out, ("model", "threads", "one_thread_duty", "rep"))
    targets = ["resnet18-v1-7", "resnet50-v1-7", "googlenet-12",
               "squeezenet1.1-7"]
    set_freq(2400000)
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            for name in targets:
                path, res = C.MODELS[name]
                # one-thread service time sets the period for BOTH arms
                s1 = _sess(path, 1)
                iname, shape = input_spec(s1, res)
                x = np.random.randn(*shape).astype(np.float32)
                warm = []
                t = time.perf_counter()
                while time.perf_counter() - t < 3.0:
                    a = time.perf_counter()
                    s1.run(None, {iname: x})
                    warm.append(time.perf_counter() - a)
                warm.sort()
                svc1 = warm[len(warm) // 2]
                del s1
                for d1 in ONE_THREAD_DUTY:
                    period = svc1 / d1
                    for th in (1, 4):
                        key = (name, str(th), str(d1), str(rep))
                        if key in seen:
                            continue
                        cooldown(C.COOL_TO)
                        sess = _sess(path, th)
                        iname, shape = input_spec(sess, res)
                        x = np.random.randn(*shape).astype(np.float32)
                        t = time.perf_counter()
                        while time.perf_counter() - t < 2.0:
                            sess.run(None, {iname: x})
                        lat, over = [], 0
                        t0 = time.perf_counter()
                        nxt = t0
                        while time.perf_counter() - t0 < WINDOW:
                            nxt += period
                            a = time.perf_counter()
                            sess.run(None, {iname: x})
                            b = time.perf_counter()
                            lat.append((b - a) * 1000.0)
                            if b < nxt:
                                time.sleep(nxt - b)
                            else:
                                over += 1
                        t1 = time.perf_counter()
                        lat.sort()
                        rec = {"stage": "duty3", "model": name, "threads": th,
                               "one_thread_duty": d1, "rep": rep,
                               "period_ms": 1000.0 * period,
                               "svc1_ms": 1000.0 * svc1,
                               "window_s": t1 - t0,
                               "lat_median_ms": lat[len(lat) // 2],
                               "n_inferences": len(lat), "n_overruns": over,
                               "duty_cycle": sum(lat) / 1000.0 / (t1 - t0),
                               "freq_khz": 2400000,
                               "freq_actual_khz": cur_freq(),
                               "temp_end_c": soc_temp_c()}
                        e = pmic.integrate(t0, t1)
                        if e:
                            rec.update(e)
                            rec["energy_per_inf_mJ"] = \
                                1000.0 * e["energy_total_J"] / len(lat)
                            rec["window_energy_J"] = e["energy_total_J"]
                        del sess
                        append(out, rec)
                        print("[duty3] %-18s t=%d d1=%.2f T=%.0fms r%d n=%-4d "
                              "duty=%.2f P=%.2f E/inf=%.1f over=%d"
                              % (name, th, d1, 1000 * period, rep,
                                 len(lat), rec["duty_cycle"],
                                 rec.get("P_mean_W", -1),
                                 rec.get("energy_per_inf_mJ", -1), over),
                              flush=True)
            if ("__idle__", "0", "0", str(rep)) not in seen:
                cooldown(C.COOL_TO)
                t0 = time.perf_counter()
                time.sleep(30.0)
                t1 = time.perf_counter()
                rec = {"stage": "duty3", "model": "__idle__", "threads": 0,
                       "one_thread_duty": 0, "rep": rep, "freq_khz": 2400000,
                       "temp_end_c": soc_temp_c()}
                e = pmic.integrate(t0, t1)
                if e:
                    rec.update(e)
                append(out, rec)
                print("[duty3] idle r%d P=%.3f W"
                      % (rep, rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


STAGES = {"duty3": stage_duty3}

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
