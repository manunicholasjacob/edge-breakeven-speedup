#!/usr/bin/env python3
"""Shared ONNX Runtime benchmark core for the revision campaign.

Adds over the first submission's harness:
  * explicit input-shape override, so symbolic-dimension models are measured at
    a stated resolution instead of silently collapsing to 1x3x1x1,
  * an idle-power measurement mode, so the static baseline is measured at each
    operating point rather than assumed from one constant,
  * arbitrary thread counts (the first submission measured 1, 2 and 4 only),
  * optional taskset pinning so thread placement is explicit.
"""

import json
import os
import time

import numpy as np
import onnxruntime as ort

from pmic import PmicSampler, cur_freq, soc_temp_c

ort.set_default_logger_severity(3)


def make_session(path, threads, profile=False):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile:
        so.enable_profiling = True
    return ort.InferenceSession(path, sess_options=so,
                                providers=["CPUExecutionProvider"])


def input_spec(sess, res=None, batch=1):
    inp = sess.get_inputs()[0]
    shape = []
    for i, d in enumerate(inp.shape):
        if isinstance(d, int) and d > 0:
            shape.append(d)
        elif i == 0:
            shape.append(batch)
        else:
            shape.append(res if res else 224)
    # NCHW vs NHWC: place the resolution on the two spatial axes only
    if res and len(shape) == 4:
        if shape[1] in (1, 3):          # NCHW
            shape[2] = shape[3] = res
        elif shape[3] in (1, 3):        # NHWC
            shape[1] = shape[2] = res
    if batch != 1 and len(shape) == 4:
        shape[0] = batch
    return inp.name, shape


def run_cell(path, threads, meas_s=6.0, warm_s=2.0, pmic=None, res=None,
             batch=1):
    sess = make_session(path, threads)
    name, shape = input_spec(sess, res, batch)
    x = np.random.randn(*shape).astype(np.float32)

    t = time.perf_counter()
    while time.perf_counter() - t < warm_s:
        sess.run(None, {name: x})

    t0 = time.perf_counter()
    lat, n = [], 0
    while time.perf_counter() - t0 < meas_s:
        a = time.perf_counter()
        sess.run(None, {name: x})
        lat.append((time.perf_counter() - a) * 1000.0)
        n += 1
    t1 = time.perf_counter()
    lat.sort()
    span = t1 - t0
    out = {"lat_median_ms": lat[len(lat) // 2],
           "lat_p95_ms": lat[int(0.95 * (len(lat) - 1))],
           "lat_mean_ms": sum(lat) / len(lat),
           "n_inferences": n, "wall_s": span, "throughput_ips": n / span,
           "input_shape": shape, "threads": threads,
           "freq_actual_khz": cur_freq()}
    if pmic is not None:
        e = pmic.integrate(t0, t1)
        if e:
            out.update(e)
            out["energy_per_inf_mJ"] = 1000.0 * e["energy_total_J"] / n
    del sess
    return out


def measure_idle(pmic, seconds=20.0):
    """Idle board power at the current operating point, nothing running."""
    time.sleep(1.0)
    t0 = time.perf_counter()
    time.sleep(seconds)
    t1 = time.perf_counter()
    e = pmic.integrate(t0, t1)
    return e or {}


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
