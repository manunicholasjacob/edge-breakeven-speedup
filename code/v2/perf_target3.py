#!/usr/bin/env python3
"""Counter target with an explicit startup phase and a spin-wait switch.

Same two-phase protocol as perf_target2.py, with the runtime's spin-waiting
thread pool under our control. Running the pair with spinning on and off
separates instructions the thread pool retires while waiting from instructions
the model retires.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import onnxruntime as ort

from bench import input_spec

ort.set_default_logger_severity(3)

path = sys.argv[1]
threads = int(sys.argv[2])
res = int(sys.argv[3]) or None
n_iter = int(sys.argv[4])
spin = sys.argv[5]

so = ort.SessionOptions()
so.intra_op_num_threads = threads
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.add_session_config_entry("session.intra_op.allow_spinning", spin)
sess = ort.InferenceSession(path, sess_options=so,
                            providers=["CPUExecutionProvider"])

name, shape = input_spec(sess, res)
x = np.random.randn(*shape).astype(np.float32)

for _ in range(10):                      # warm-up, counted in both phases
    sess.run(None, {name: x})

t0 = time.perf_counter()
for _ in range(n_iter):
    sess.run(None, {name: x})
t1 = time.perf_counter()

print(json.dumps({"n_iter": n_iter, "wall_s": t1 - t0,
                  "ms_per_inf": (1000.0 * (t1 - t0) / n_iter) if n_iter else 0.0,
                  "shape": shape, "threads": threads, "spin": spin}))
