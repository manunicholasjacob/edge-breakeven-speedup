#!/usr/bin/env python3
"""Dense GEMM through NumPy/OpenBLAS.

A second implementation of a second workload: if the break-even rule is a
property of the board rather than of ONNX Runtime's threading, a BLAS GEMM
should obey it too. Thread count comes from the BLAS environment variables set
by the caller.
"""

import json
import sys
import time

import numpy as np

n = int(sys.argv[1])
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

a = np.random.randn(n, n).astype(np.float32)
b = np.random.randn(n, n).astype(np.float32)

t = time.perf_counter()
while time.perf_counter() - t < 1.0:
    a @ b

lat = []
t0 = time.perf_counter()
while time.perf_counter() - t0 < seconds:
    s = time.perf_counter()
    a @ b
    lat.append((time.perf_counter() - s) * 1000.0)
t1 = time.perf_counter()

lat.sort()
flops = 2.0 * n ** 3
print(json.dumps({"size": n, "n_iter": len(lat), "wall_s": t1 - t0,
                  "ms_median": lat[len(lat) // 2],
                  "gflops": flops / (lat[len(lat) // 2] / 1000.0) / 1e9}))
