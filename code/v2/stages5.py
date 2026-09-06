#!/usr/bin/env python3
"""Fifth round: does the verdict survive with the spin-wait switched off?

`perfspin` established that 94% of the growth in instructions retired per
inference is the runtime's spin-waiting thread pool, and that turning the pool
off costs only 1% of four-thread latency for ResNet-18. That raises the
question the paper has to answer and had not: if a one-line runtime setting
removes almost all of the added work, does threading then clear the dynamic
break-even bar?

  spinE  the full thread sweep, one to four threads on the whole model set, at
         2400 MHz, with the spin-wait on and off, measuring power as well as
         latency, so the break-even comparison can be recomputed under both.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign as C
from bench import append, input_spec
from pmic import PmicSampler, cooldown, cur_freq, set_freq, soc_temp_c

WINDOW = 6.0
WARM = 2.0


def stage_spinE(out, reps=3):
    import numpy as np
    import onnxruntime as ort

    seen = C.done_keys(out, ("model", "threads", "spin", "rep"))
    names = C.CNN8 if hasattr(C, "CNN8") else [
        "googlenet-12", "resnet18-v1-7", "efficientnet-lite4-11",
        "squeezenet1.1-7", "shufflenet-v2-10", "resnet50-v1-7",
        "mobilenetv2-12", "densenet-12"]
    names = names + ["mnv3-cifar"]
    pmic = PmicSampler()
    pmic.start()
    set_freq(2400000)
    try:
        for rep in range(1, reps + 1):
            for spin in ("1", "0"):
                for th in (1, 2, 3, 4):
                    for name in names:
                        if (name, str(th), spin, str(rep)) in seen:
                            continue
                        path, res = C.MODELS[name]
                        cooldown(C.COOL_TO)
                        so = ort.SessionOptions()
                        so.intra_op_num_threads = th
                        so.inter_op_num_threads = 1
                        so.graph_optimization_level = \
                            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        so.add_session_config_entry(
                            "session.intra_op.allow_spinning", spin)
                        sess = ort.InferenceSession(
                            path, sess_options=so,
                            providers=["CPUExecutionProvider"])
                        iname, shape = input_spec(sess, res)
                        x = np.random.randn(*shape).astype(np.float32)
                        t = time.perf_counter()
                        while time.perf_counter() - t < WARM:
                            sess.run(None, {iname: x})
                        lat = []
                        t0 = time.perf_counter()
                        while time.perf_counter() - t0 < WINDOW:
                            a = time.perf_counter()
                            sess.run(None, {iname: x})
                            lat.append((time.perf_counter() - a) * 1000.0)
                        t1 = time.perf_counter()
                        lat.sort()
                        rec = {"stage": "spinE", "model": name, "threads": th,
                               "spin": spin, "rep": rep,
                               "lat_median_ms": lat[len(lat) // 2],
                               "lat_p95_ms": lat[int(0.95 * (len(lat) - 1))],
                               "n_inferences": len(lat),
                               "freq_khz": 2400000,
                               "freq_actual_khz": cur_freq(),
                               "temp_before_c": soc_temp_c()}
                        e = pmic.integrate(t0, t1)
                        if e:
                            rec.update(e)
                            rec["energy_per_inf_mJ"] = \
                                1000.0 * e["energy_total_J"] / len(lat)
                        append(out, rec)
                        print("[spinE] %-22s t=%d spin=%s r%d %8.2f ms "
                              "P=%.2f W" % (name, th, spin, rep,
                                            rec["lat_median_ms"],
                                            rec.get("P_mean_W", -1)),
                              flush=True)
                        del sess
    finally:
        pmic.stop()


def stage_idleE(out, reps=3):
    """The static floor again, in the same session as spinE.

    The dynamic threshold subtracts the floor, so a floor measured in a
    different campaign imports that campaign's offset into the comparison. This
    measures it alongside spinE so the recomputation is self-contained.
    """
    seen = C.done_keys(out, ("rep",))
    set_freq(2400000)
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            if (str(rep),) in seen:
                continue
            cooldown(C.COOL_TO)
            t0 = time.perf_counter()
            time.sleep(30.0)
            t1 = time.perf_counter()
            rec = {"stage": "idleE", "rep": rep, "freq_khz": 2400000,
                   "freq_actual_khz": cur_freq(),
                   "temp_before_c": soc_temp_c()}
            e = pmic.integrate(t0, t1)
            if e:
                rec.update(e)
            append(out, rec)
            print("[idleE] r%d P=%.3f W" % (rep, rec.get("P_mean_W", -1)),
                  flush=True)
    finally:
        pmic.stop()


STAGES = {"spinE": stage_spinE, "idleE": stage_idleE}

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
