#!/usr/bin/env python3
"""Revision campaign for the break-even parallel speedup study.

Stages, each answering a specific reviewer objection:

  idle       static board power measured at every operating point, so the
             static/dynamic split is measured rather than assumed  (R1)
  threads    1..4 threads on eight architectures at two clocks, the full
             scalability sweep                                      (R3)
  perf       ARM PMU counters per thread count: cache refills, bus traffic,
             stalled cycles, so the speedup is explained not asserted (R1, R3)
  ops        per-operator ONNX Runtime profiles at three resolutions, which
             is where the parallelism actually lives                 (R3)
  gemm       a pure GEMM workload through a different library, so the result
             is not an artifact of one runtime's threading           (R1)
  llm        llama.cpp decode thread sweep, a memory-bound workload  (R1)
  cotenant   the co-tenancy claim the first submission asserted      (R3)
  sustained  five-minute runs, to test the six-second protocol       (R1)
  governor   ondemand/schedutil against the pinned clock             (R2)

Every stage appends JSONL and skips cells already recorded, so the campaign is
resumable after an interruption.
"""

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench import append, measure_idle, run_cell
from pmic import (PmicSampler, cooldown, cur_freq, governor, limits, set_freq,
                  soc_temp_c, write_limits)

HOME = os.path.expanduser("~")
M = os.path.join(HOME, "p15v2", "models")

# name -> (path, resolution override or None)
MODELS = {
    "squeezenet1.1-7":       (os.path.join(HOME, "memwall/models/squeezenet1.1-7.onnx"), None),
    "shufflenet-v2-10":      (os.path.join(HOME, "memwall/models/shufflenet-v2-10.onnx"), None),
    "googlenet-12":          (os.path.join(M, "googlenet-12.onnx"), None),
    "efficientnet-lite4-11": (os.path.join(M, "efficientnet-lite4-11.onnx"), None),
    "mobilenetv2-12":        (os.path.join(HOME, "p15res/mobilenetv2-12__dynspatial.onnx"), 224),
    "resnet18-v1-7":         (os.path.join(HOME, "coldstart-llm/data/resnet18-v1-7__dynspatial.onnx"), 224),
    "resnet50-v1-7":         (os.path.join(HOME, "p15res/resnet50-v1-7__dynspatial.onnx"), 224),
    "densenet-12":           (os.path.join(HOME, "p15res/densenet-12__dynspatial.onnx"), 224),
    "mnv3-cifar":            (os.path.join(HOME, "Desktop/Researchpaper8/models/mnv3_c100_final_fp32.onnx"), None),
}
MNV3 = os.path.join(HOME, "Desktop/Researchpaper8/models/mnv3_c100_final_fp32.onnx")
RES_MODEL = os.path.join(HOME, "coldstart-llm/data/resnet18-v1-7__dynspatial.onnx")

CLOCKS = [2400000, 1500000]
ALL_CLOCKS = [1500000, 1800000, 2100000, 2400000]
THREADS = [1, 2, 3, 4]
COOL_TO = 52.0


def done_keys(path, fields):
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r:
                continue
            seen.add(tuple(str(r.get(k)) for k in fields))
    return seen


# ------------------------------------------------------------------ idle ----
def stage_idle(out, reps=3):
    seen = done_keys(out, ("freq_khz", "rep", "mode"))
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            for khz in ALL_CLOCKS:
                for mode in ("idle", "idle_after_load"):
                    key = (str(khz), str(rep), mode)
                    if key in seen:
                        continue
                    set_freq(khz)
                    if mode == "idle_after_load":
                        # 20 s of four-thread load, then measure the decay floor
                        subprocess.run(["stress-ng", "--cpu", "4", "--timeout", "20s"],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, check=False)
                    cooldown(COOL_TO)
                    rec = {"stage": "idle", "freq_khz": khz, "rep": rep,
                           "mode": mode, "temp_before_c": soc_temp_c(),
                           "freq_actual_khz": cur_freq(),
                           "governor": governor()}
                    rec.update(measure_idle(pmic, 30.0))
                    rec["temp_after_c"] = soc_temp_c()
                    append(out, rec)
                    print("[idle] %d MHz rep%d %s P=%.3f W" %
                          (khz // 1000, rep, mode, rec.get("P_mean_W", -1)),
                          flush=True)
    finally:
        pmic.stop()


# --------------------------------------------------------------- threads ----
def stage_threads(out, reps=3):
    seen = done_keys(out, ("model", "freq_khz", "threads", "rep"))
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            for khz in CLOCKS:
                set_freq(khz)
                for th in THREADS:
                    for name, (path, res) in MODELS.items():
                        key = (name, str(khz), str(th), str(rep))
                        if key in seen:
                            continue
                        if not os.path.exists(path):
                            append(out, {"stage": "threads", "model": name,
                                         "freq_khz": khz, "threads": th,
                                         "rep": rep, "error": "missing model"})
                            continue
                        cooldown(COOL_TO)
                        rec = {"stage": "threads", "model": name,
                               "freq_khz": khz, "rep": rep,
                               "temp_before_c": soc_temp_c()}
                        try:
                            rec.update(run_cell(path, th, pmic=pmic, res=res))
                        except Exception as e:
                            rec["error"] = str(e)[:200]
                        rec["temp_after_c"] = soc_temp_c()
                        append(out, rec)
                        print("[threads] %4dMHz t=%d %-22s r%d %8.3f ms %7.2f mJ P=%.2f W"
                              % (khz // 1000, th, name, rep,
                                 rec.get("lat_median_ms", -1),
                                 rec.get("energy_per_inf_mJ", -1),
                                 rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


# ------------------------------------------------------------------ perf ----
PERF_EVENTS = ["cycles", "instructions", "cache-references", "cache-misses",
               "stalled-cycles-backend", "l1d_cache", "l1d_cache_refill",
               "l2d_cache", "l2d_cache_refill", "bus_access", "mem_access"]


def stage_perf(out, reps=2):
    seen = done_keys(out, ("model", "res", "threads", "rep"))
    targets = [("resnet18-v1-7", RES_MODEL, 224),
               ("resnet18-v1-7", RES_MODEL, 32),
               ("resnet18-v1-7", RES_MODEL, 96),
               ("mnv3-cifar", MNV3, None),
               ("squeezenet1.1-7", MODELS["squeezenet1.1-7"][0], None),
               ("densenet-12", MODELS["densenet-12"][0], 224)]
    runner = os.path.join(HOME, "p15v2", "perf_target.py")
    set_freq(2400000)
    for rep in range(1, reps + 1):
        for name, path, res in targets:
            for th in THREADS:
                key = (name, str(res), str(th), str(rep))
                if key in seen or not os.path.exists(path):
                    continue
                cooldown(COOL_TO)
                cmd = ["sudo", "perf", "stat", "-x,", "-e", ",".join(PERF_EVENTS),
                       "--", sys.executable, runner, path, str(th), str(res or 0)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                counters = {}
                for line in p.stderr.splitlines():
                    parts = line.split(",")
                    if len(parts) >= 3 and parts[0] not in ("", "<not counted>"):
                        try:
                            counters[parts[2]] = float(parts[0])
                        except ValueError:
                            pass
                body = {}
                for line in p.stdout.splitlines():
                    if line.startswith("{"):
                        try:
                            body = json.loads(line)
                        except Exception:
                            pass
                rec = {"stage": "perf", "model": name, "res": res, "threads": th,
                       "rep": rep, "counters": counters, "run": body,
                       "freq_actual_khz": cur_freq()}
                append(out, rec)
                print("[perf] %-16s res=%s t=%d r%d ipc=%.2f l2_refill=%.3g"
                      % (name, res, th, rep,
                         counters.get("instructions", 0) / max(counters.get("cycles", 1), 1),
                         counters.get("l2d_cache_refill", 0)), flush=True)


# ------------------------------------------------------------------- ops ----
def stage_ops(out):
    import onnxruntime as ort  # noqa: F401
    from bench import input_spec, make_session
    import numpy as np
    seen = done_keys(out, ("model", "res", "threads"))
    set_freq(2400000)
    for res in (32, 96, 224):
        for th in (1, 4):
            key = ("resnet18-v1-7", str(res), str(th))
            if key in seen:
                continue
            sess = make_session(RES_MODEL, th, profile=True)
            name, shape = input_spec(sess, res)
            x = np.random.randn(*shape).astype("float32")
            for _ in range(30):
                sess.run(None, {name: x})
            prof = sess.end_profiling()
            with open(prof) as f:
                events = json.load(f)
            agg = {}
            for e in events:
                if e.get("cat") == "Node" and "dur" in e:
                    op = e.get("args", {}).get("op_name", "?")
                    a = agg.setdefault(op, {"n": 0, "us": 0.0})
                    a["n"] += 1
                    a["us"] += e["dur"]
            os.remove(prof)
            rec = {"stage": "ops", "model": "resnet18-v1-7", "res": res,
                   "threads": th, "shape": shape, "ops": agg}
            append(out, rec)
            top = sorted(agg.items(), key=lambda kv: -kv[1]["us"])[:3]
            print("[ops] res=%d t=%d top=%s" % (res, th, [(k, round(v['us'])) for k, v in top]),
                  flush=True)
            del sess


# ------------------------------------------------------------------ gemm ----
def stage_gemm(out, reps=3):
    seen = done_keys(out, ("size", "threads", "rep"))
    runner = os.path.join(HOME, "p15v2", "gemm_target.py")
    pmic = PmicSampler()
    pmic.start()
    set_freq(2400000)
    try:
        for rep in range(1, reps + 1):
            for size in (128, 256, 512, 1024):
                for th in THREADS:
                    key = (str(size), str(th), str(rep))
                    if key in seen:
                        continue
                    cooldown(COOL_TO)
                    env = dict(os.environ)
                    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                        env[var] = str(th)
                    t0 = time.perf_counter()
                    p = subprocess.run([sys.executable, runner, str(size), "6.0"],
                                       capture_output=True, text=True, env=env)
                    t1 = time.perf_counter()
                    body = {}
                    for line in p.stdout.splitlines():
                        if line.startswith("{"):
                            body = json.loads(line)
                    rec = {"stage": "gemm", "size": size, "threads": th,
                           "rep": rep, "freq_actual_khz": cur_freq()}
                    rec.update(body)
                    e = pmic.integrate(t0, t1)
                    if e:
                        rec.update(e)
                        if body.get("n_iter"):
                            rec["energy_per_gemm_mJ"] = (
                                1000.0 * e["energy_total_J"] / body["n_iter"])
                    append(out, rec)
                    print("[gemm] n=%d t=%d r%d %.3f ms P=%.2f W"
                          % (size, th, rep, body.get("ms_median", -1),
                             rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


# ------------------------------------------------------------------- llm ----
def stage_llm(out, reps=3):
    seen = done_keys(out, ("model", "threads", "rep"))
    bench_bin = os.path.join(HOME, "llm/llama.cpp/build/bin/llama-bench")
    gguf = os.path.join(HOME, "llm/models/qwen0.5b-q4km.gguf")
    if not (os.path.exists(bench_bin) and os.path.exists(gguf)):
        print("[llm] binaries or model missing, skipping", flush=True)
        return
    pmic = PmicSampler()
    pmic.start()
    set_freq(2400000)
    try:
        for rep in range(1, reps + 1):
            for th in THREADS:
                key = ("qwen0.5b-q4km", str(th), str(rep))
                if key in seen:
                    continue
                cooldown(COOL_TO)
                t0 = time.perf_counter()
                p = subprocess.run([bench_bin, "-m", gguf, "-t", str(th),
                                    "-p", "0", "-n", "64", "-r", "2"],
                                   capture_output=True, text=True)
                t1 = time.perf_counter()
                tg = None
                for line in p.stdout.splitlines():
                    m = re.search(r"tg64\s*\|\s*([0-9.]+)", line)
                    if m:
                        tg = float(m.group(1))
                    elif "tg64" in line:
                        nums = re.findall(r"([0-9]+\.[0-9]+)", line)
                        if nums:
                            tg = float(nums[-2]) if len(nums) > 1 else float(nums[-1])
                rec = {"stage": "llm", "model": "qwen0.5b-q4km", "threads": th,
                       "rep": rep, "tokens_per_s": tg, "wall_s": t1 - t0,
                       "freq_actual_khz": cur_freq(),
                       "stdout_tail": p.stdout[-400:]}
                e = pmic.integrate(t0, t1)
                if e:
                    rec.update(e)
                    if tg:
                        rec["energy_per_token_mJ"] = 1000.0 * rec["P_mean_W"] / tg
                append(out, rec)
                print("[llm] t=%d r%d %.2f tok/s P=%.2f W" %
                      (th, rep, tg or -1, rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


# -------------------------------------------------------------- cotenant ----
def stage_cotenant(out, reps=3):
    seen = done_keys(out, ("cond", "rep"))
    pmic = PmicSampler()
    pmic.start()
    set_freq(2400000)
    path, res = MODELS["resnet18-v1-7"]
    try:
        for rep in range(1, reps + 1):
            for cond in ("infer4_alone", "infer1_alone", "infer1_plus_3cotenant",
                         "cotenant4_alone"):
                key = (cond, str(rep))
                if key in seen:
                    continue
                cooldown(COOL_TO)
                rec = {"stage": "cotenant", "cond": cond, "rep": rep,
                       "temp_before_c": soc_temp_c()}
                hog = None
                bogo_before = None
                if cond == "infer1_plus_3cotenant":
                    hog = subprocess.Popen(
                        ["stress-ng", "--cpu", "3", "--timeout", "30s",
                         "--metrics-brief"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True)
                    time.sleep(2.0)
                if cond == "cotenant4_alone":
                    t0 = time.perf_counter()
                    p = subprocess.run(["stress-ng", "--cpu", "4", "--timeout", "8s",
                                        "--metrics-brief"],
                                       capture_output=True, text=True)
                    t1 = time.perf_counter()
                    m = re.search(r"cpu\s+(\d+)\s", p.stdout + p.stderr)
                    rec["cotenant_bogo_ops"] = int(m.group(1)) if m else None
                    e = pmic.integrate(t0, t1)
                    if e:
                        rec.update(e)
                else:
                    th = 1 if cond.startswith("infer1") else 4
                    try:
                        rec.update(run_cell(path, th, pmic=pmic, res=res))
                    except Exception as exc:
                        rec["error"] = str(exc)[:200]
                if hog is not None:
                    try:
                        o = hog.communicate(timeout=40)[0]
                        m = re.search(r"cpu\s+(\d+)\s", o or "")
                        rec["cotenant_bogo_ops"] = int(m.group(1)) if m else None
                    except Exception:
                        hog.kill()
                rec["temp_after_c"] = soc_temp_c()
                append(out, rec)
                print("[cotenant] %-22s r%d %8.3f ms P=%.2f W bogo=%s"
                      % (cond, rep, rec.get("lat_median_ms", -1),
                         rec.get("P_mean_W", -1), rec.get("cotenant_bogo_ops")),
                      flush=True)
    finally:
        pmic.stop()


# ------------------------------------------------------------- sustained ----
def stage_sustained(out, seconds=300.0):
    seen = done_keys(out, ("model", "threads"))
    pmic = PmicSampler()
    pmic.start()
    set_freq(2400000)
    try:
        for name, path, res in (("resnet18-v1-7", RES_MODEL, 224),
                                ("mnv3-cifar", MNV3, None)):
            for th in (1, 4):
                key = (name, str(th))
                if key in seen or not os.path.exists(path):
                    continue
                cooldown(COOL_TO)
                rec = {"stage": "sustained", "model": name, "threads": th,
                       "seconds": seconds, "temp_before_c": soc_temp_c()}
                try:
                    rec.update(run_cell(path, th, meas_s=seconds, warm_s=3.0,
                                        pmic=pmic, res=res))
                except Exception as e:
                    rec["error"] = str(e)[:200]
                rec["temp_after_c"] = soc_temp_c()
                append(out, rec)
                print("[sustained] %-16s t=%d %.3f ms P=%.2f W temp %.1f->%.1f"
                      % (name, th, rec.get("lat_median_ms", -1),
                         rec.get("P_mean_W", -1), rec["temp_before_c"],
                         rec["temp_after_c"]), flush=True)
    finally:
        pmic.stop()


# -------------------------------------------------------------- governor ----
def stage_governor(out, reps=3):
    seen = done_keys(out, ("gov", "model", "threads", "rep"))
    lo, hi = limits()
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            for gov in ("ondemand", "schedutil", "performance"):
                for name in ("resnet18-v1-7", "mobilenetv2-12"):
                    path, res = MODELS[name]
                    for th in (1, 4):
                        key = (gov, name, str(th), str(rep))
                        if key in seen:
                            continue
                        write_limits(lo, hi)
                        got = governor(gov)
                        cooldown(COOL_TO)
                        freqs = []

                        rec = {"stage": "governor", "gov": gov, "gov_actual": got,
                               "model": name, "threads": th, "rep": rep,
                               "temp_before_c": soc_temp_c()}
                        try:
                            rec.update(run_cell(path, th, pmic=pmic, res=res))
                        except Exception as e:
                            rec["error"] = str(e)[:200]
                        rec["freq_end_khz"] = cur_freq()
                        rec["temp_after_c"] = soc_temp_c()
                        append(out, rec)
                        print("[gov] %-11s %-16s t=%d r%d %8.3f ms P=%.2f W f=%d"
                              % (gov, name, th, rep, rec.get("lat_median_ms", -1),
                                 rec.get("P_mean_W", -1), rec["freq_end_khz"]),
                              flush=True)
    finally:
        pmic.stop()
        governor("performance")


# --------------------------------------------------------------- idlegov ----
def stage_idlegov(out, reps=3):
    """The static floor a real deployment actually idles at.

    The `idle` stage pins the clock, which answers what the floor is at each
    operating point. A deployed board idles under a governor and drops to the
    bottom of the range, so the floor that belongs in Equation (2) may be lower
    than the floor at the clock the inference runs at. This stage measures it.
    """
    seen = done_keys(out, ("mode", "rep"))
    lo, hi = limits()
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in range(1, reps + 1):
            for mode in ("ondemand", "schedutil", "pinned_min", "pinned_max"):
                key = (mode, str(rep))
                if key in seen:
                    continue
                if mode in ("ondemand", "schedutil"):
                    write_limits(lo, hi)
                    got = governor(mode)
                else:
                    governor("performance")
                    set_freq(1500000 if mode == "pinned_min" else 2400000)
                    got = "performance"
                cooldown(COOL_TO)
                rec = {"stage": "idlegov", "mode": mode, "rep": rep,
                       "governor_actual": got, "temp_before_c": soc_temp_c()}
                rec.update(measure_idle(pmic, 30.0))
                rec["freq_after_khz"] = cur_freq()
                rec["temp_after_c"] = soc_temp_c()
                append(out, rec)
                print("[idlegov] %-12s rep%d P=%.3f W f=%d" %
                      (mode, rep, rec.get("P_mean_W", -1),
                       rec["freq_after_khz"]), flush=True)
    finally:
        pmic.stop()
        governor("performance")


# ------------------------------------------------------------------ spin ----
def stage_spin(out, reps=3):
    """Why three-thread cells are bimodal on a four-core board.

    The thread sweep showed every three-thread block at both clocks containing
    one repetition that was slower and drew less power than the other two, in
    every model of the block, while one, two and four-thread cells varied by
    under one percent. The suspected mechanism is the runtime's spin-waiting
    intra-op thread pool: with three worker threads plus the sampler and the
    main thread on four cores, occupancy is borderline and placement decides.
    Disabling spinning removes the spin-wait if that is the cause.
    """
    import onnxruntime as ort
    import numpy as np
    from bench import input_spec

    seen = done_keys(out, ("model", "threads", "spin", "rep"))
    targets = ["resnet18-v1-7", "googlenet-12", "squeezenet1.1-7"]
    pmic = PmicSampler()
    pmic.start()
    set_freq(2400000)
    try:
        for rep in range(1, reps + 1):
            for spin in ("1", "0"):
                for th in (2, 3, 4):
                    for name in targets:
                        key = (name, str(th), spin, str(rep))
                        if key in seen:
                            continue
                        path, res = MODELS[name]
                        cooldown(COOL_TO)
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
                        t0 = time.perf_counter()
                        while time.perf_counter() - t0 < 2.0:
                            sess.run(None, {iname: x})
                        lat, n = [], 0
                        t0 = time.perf_counter()
                        while time.perf_counter() - t0 < 6.0:
                            a = time.perf_counter()
                            sess.run(None, {iname: x})
                            lat.append((time.perf_counter() - a) * 1000.0)
                            n += 1
                        t1 = time.perf_counter()
                        lat.sort()
                        rec = {"stage": "spin", "model": name, "threads": th,
                               "spin": spin, "rep": rep,
                               "lat_median_ms": lat[len(lat) // 2],
                               "lat_p95_ms": lat[int(0.95 * (len(lat) - 1))],
                               "n_inferences": n,
                               "freq_actual_khz": cur_freq(),
                               "temp_before_c": soc_temp_c()}
                        e = pmic.integrate(t0, t1)
                        if e:
                            rec.update(e)
                            rec["energy_per_inf_mJ"] = \
                                1000.0 * e["energy_total_J"] / n
                        append(out, rec)
                        print("[spin] %-18s t=%d spin=%s r%d %8.2f ms P=%.2f W"
                              % (name, th, spin, rep, rec["lat_median_ms"],
                                 rec.get("P_mean_W", -1)), flush=True)
                        del sess
    finally:
        pmic.stop()


# -------------------------------------------------------------- threads3 ----
def stage_threads3(out, reps=(4, 5, 6)):
    """Extra repetitions for the three-thread cells only, so their medians rest
    on the same footing as the rest of the sweep."""
    seen = done_keys(out, ("model", "freq_khz", "threads", "rep"))
    pmic = PmicSampler()
    pmic.start()
    try:
        for rep in reps:
            for khz in CLOCKS:
                set_freq(khz)
                for name, (path, res) in MODELS.items():
                    key = (name, str(khz), "3", str(rep))
                    if key in seen or not os.path.exists(path):
                        continue
                    cooldown(COOL_TO)
                    rec = {"stage": "threads", "model": name, "freq_khz": khz,
                           "rep": rep, "temp_before_c": soc_temp_c()}
                    try:
                        rec.update(run_cell(path, 3, pmic=pmic, res=res))
                    except Exception as e:
                        rec["error"] = str(e)[:200]
                    rec["temp_after_c"] = soc_temp_c()
                    append(out, rec)
                    print("[threads3] %4dMHz %-22s r%d %8.2f ms P=%.2f W"
                          % (khz // 1000, name, rep,
                             rec.get("lat_median_ms", -1),
                             rec.get("P_mean_W", -1)), flush=True)
    finally:
        pmic.stop()


STAGES = {"idle": stage_idle, "threads": stage_threads, "perf": stage_perf,
          "ops": stage_ops, "gemm": stage_gemm, "llm": stage_llm,
          "cotenant": stage_cotenant, "sustained": stage_sustained,
          "governor": stage_governor, "idlegov": stage_idlegov, "spin": stage_spin,
          "threads3": stage_threads3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo, hi = limits()
    gov0 = governor()
    atexit.register(write_limits, lo, hi)
    atexit.register(governor, gov0)
    print("# stage=%s out=%s (will restore %s-%s, governor %s)"
          % (args.stage, args.out, lo, hi, gov0), flush=True)
    t0 = time.time()
    STAGES[args.stage](args.out)
    print("# stage %s complete in %.1f min" % (args.stage, (time.time() - t0) / 60.0),
          flush=True)


if __name__ == "__main__":
    main()
