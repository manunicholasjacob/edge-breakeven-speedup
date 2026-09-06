#!/usr/bin/env python3
"""Fixed-work inference target, run under `perf stat`.

Counters must cover the same amount of work at every thread count, so this runs
a fixed number of inferences rather than a fixed time window.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from bench import input_spec, make_session

path = sys.argv[1]
threads = int(sys.argv[2])
res = int(sys.argv[3]) or None
n_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 60

sess = make_session(path, threads)
name, shape = input_spec(sess, res)
x = np.random.randn(*shape).astype(np.float32)

for _ in range(10):
    sess.run(None, {name: x})

t0 = time.perf_counter()
for _ in range(n_iter):
    sess.run(None, {name: x})
t1 = time.perf_counter()

print(json.dumps({"n_iter": n_iter, "wall_s": t1 - t0,
                  "ms_per_inf": 1000.0 * (t1 - t0) / n_iter,
                  "shape": shape, "threads": threads}))
