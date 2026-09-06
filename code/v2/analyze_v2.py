#!/usr/bin/env python3
"""Analysis for the revision campaign.

Reads the E_*.jsonl files produced by campaign.py and prints, for each reviewer
objection, the table that answers it. Every number quoted in the revised
manuscript comes from here.
"""

import argparse
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DEFAULT = os.path.join(HERE, "..", "..", "data", "v2")
V1 = os.path.join(HERE, "..", "..", "data")

MODEL_ORDER = ["googlenet-12", "resnet18-v1-7", "efficientnet-lite4-11",
               "squeezenet1.1-7", "shufflenet-v2-10", "resnet50-v1-7",
               "mobilenetv2-12", "densenet-12"]
SHORT = {"googlenet-12": "GoogLeNet", "resnet18-v1-7": "ResNet-18",
         "efficientnet-lite4-11": "EfficientNet-Lite4",
         "squeezenet1.1-7": "SqueezeNet 1.1", "shufflenet-v2-10": "ShuffleNet-v2",
         "resnet50-v1-7": "ResNet-50", "mobilenetv2-12": "MobileNetV2",
         "densenet-12": "DenseNet-121", "mnv3-cifar": "MobileNetV3 (CIFAR)"}


def load(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


# ------------------------------------------------------------------ idle ----
def idle_table(D):
    recs = [r for r in load(os.path.join(D, "E_idle.jsonl")) if "P_mean_W" in r]
    if not recs:
        return {}
    out = {}
    print("\n=== E1  measured idle (static) board power ===")
    print(f"{'clock MHz':>10} {'mode':>18} {'n':>3} {'P_idle W':>9} {'spread':>8}")
    for khz in sorted({r["freq_khz"] for r in recs}):
        for mode in ("idle", "idle_after_load"):
            v = [r["P_mean_W"] for r in recs
                 if r["freq_khz"] == khz and r["mode"] == mode]
            if not v:
                continue
            print(f"{khz//1000:>10} {mode:>18} {len(v):>3} {med(v):>9.3f} "
                  f"{max(v)-min(v):>8.3f}")
            if mode == "idle":
                out[khz] = med(v)
    if out:
        print(f"\nThe first submission assumed a single 2.063 W idle baseline at "
              f"every clock.\nMeasured: " +
              ", ".join(f"{k//1000} MHz {v:.3f} W" for k, v in sorted(out.items())))
    return out


# --------------------------------------------------------------- threads ----
def thread_tables(D, p_idle):
    recs = [r for r in load(os.path.join(D, "E_threads.jsonl"))
            if "lat_median_ms" in r and "P_mean_W" in r]
    if not recs:
        return
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        by[r["freq_khz"]][r["model"]][r["threads"]].append(r)

    for khz in sorted(by, reverse=True):
        pi = p_idle.get(khz)
        print(f"\n=== E2  thread scaling at {khz//1000} MHz "
              f"(measured idle {pi:.3f} W) ===" if pi else
              f"\n=== E2  thread scaling at {khz//1000} MHz ===")
        print(f"{'model':>20} {'t':>2} {'ms':>8} {'S_n':>6} {'P W':>6} "
              f"{'Ptot':>6} {'Pdyn':>6} {'Etot':>6} {'Edyn':>6} {'cv%':>5}")
        for m in MODEL_ORDER:
            if m not in by[khz]:
                continue
            d = by[khz][m]
            if 1 not in d:
                continue
            t1 = med([r["lat_median_ms"] for r in d[1]])
            p1 = med([r["P_mean_W"] for r in d[1]])
            for th in sorted(d):
                tn = med([r["lat_median_ms"] for r in d[th]])
                pn = med([r["P_mean_W"] for r in d[th]])
                lats = [r["lat_median_ms"] for r in d[th]]
                cv = (100.0 * st.pstdev(lats) / st.mean(lats)) if len(lats) > 1 else 0.0
                s = t1 / tn
                ptot = pn / p1
                etot = ptot / s
                if pi and p1 > pi and pn > pi:
                    pdyn = (pn - pi) / (p1 - pi)
                    edyn = pdyn / s
                else:
                    pdyn = edyn = float("nan")
                print(f"{SHORT.get(m, m):>20} {th:>2} {tn:>8.2f} {s:>6.2f} {pn:>6.2f} "
                      f"{ptot:>6.2f} {pdyn:>6.2f} {etot:>6.3f} {edyn:>6.3f} {cv:>5.1f}")

        # break-even thresholds
        p = {}
        for th in (1, 2, 3, 4):
            v = [r["P_mean_W"] for m in by[khz] for r in by[khz][m].get(th, [])]
            if v:
                p[th] = med(v)
        if p and pi:
            print(f"\n  break-even speedup at {khz//1000} MHz:")
            for th in sorted(p):
                if th == 1:
                    continue
                print(f"    {th} threads: total-energy {p[th]/p[1]:.2f}x   "
                      f"dynamic-energy {(p[th]-pi)/(p[1]-pi):.2f}x")


# ------------------------------------------------------------------ perf ----
def perf_table(D):
    recs = [r for r in load(os.path.join(D, "E_perf.jsonl")) if r.get("counters")]
    if not recs:
        return
    print("\n=== E3  ARM PMU counters per inference (2400 MHz) ===")
    print(f"{'target':>22} {'t':>2} {'ms/inf':>7} {'IPC':>5} {'stall%':>7} "
          f"{'L2refill/inf':>13} {'bus/inf':>11} {'MB/s':>8}")
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        key = (r["model"], r.get("res"))
        by[key][r["threads"]].append(r)
    for key in sorted(by, key=lambda k: (str(k[0]), str(k[1]))):
        for th in sorted(by[key]):
            rs = by[key][th]
            c = {k: med([r["counters"].get(k) for r in rs])
                 for k in rs[0]["counters"]}
            n = med([r["run"].get("n_iter", 0) for r in rs]) or 1
            wall = med([r["run"].get("wall_s", 0) for r in rs]) or 1
            ipc = c.get("instructions", 0) / max(c.get("cycles", 1), 1)
            stall = 100.0 * c.get("stalled-cycles-backend", 0) / max(c.get("cycles", 1), 1)
            l2 = c.get("l2d_cache_refill", 0) / n
            bus = c.get("bus_access", 0) / n
            mbs = c.get("bus_access", 0) * 64.0 / wall / 1e6
            label = f"{SHORT.get(key[0], key[0])}@{key[1] or 'native'}"
            print(f"{label:>22} {th:>2} {1000*wall/n:>7.2f} {ipc:>5.2f} {stall:>7.1f} "
                  f"{l2:>13.0f} {bus:>11.0f} {mbs:>8.0f}")


# ------------------------------------------------------------------- ops ----
def ops_table(D):
    recs = load(os.path.join(D, "E_ops.jsonl"))
    if not recs:
        return
    print("\n=== E4  where the time goes, per operator (ResNet-18, 30 runs) ===")
    for res in sorted({r["res"] for r in recs}):
        rs = {r["threads"]: r for r in recs if r["res"] == res}
        if 1 not in rs or 4 not in rs:
            continue
        a, b = rs[1]["ops"], rs[4]["ops"]
        tot1 = sum(v["us"] for v in a.values())
        tot4 = sum(v["us"] for v in b.values())
        print(f"\n  resolution {res}x{res}: total node time "
              f"{tot1/1000:.1f} ms (1 thread) -> {tot4/1000:.1f} ms (4 threads), "
              f"node-level speedup {tot1/tot4:.2f}x")
        print(f"  {'op':>18} {'n':>5} {'us@1':>9} {'us@4':>9} {'speedup':>8} {'share%':>7}")
        for op in sorted(a, key=lambda k: -a[k]["us"])[:6]:
            if op not in b:
                continue
            sp = a[op]["us"] / b[op]["us"] if b[op]["us"] else float("nan")
            print(f"  {op:>18} {a[op]['n']:>5} {a[op]['us']:>9.0f} {b[op]['us']:>9.0f} "
                  f"{sp:>8.2f} {100*a[op]['us']/tot1:>7.1f}")


# ------------------------------------------------------- gemm / llm / etc ----
def gemm_table(D, p_idle):
    recs = [r for r in load(os.path.join(D, "E_gemm.jsonl")) if "ms_median" in r]
    if not recs:
        return
    pi = p_idle.get(2400000)
    print("\n=== E5  dense GEMM through OpenBLAS (a different library, 2400 MHz) ===")
    print(f"{'size':>6} {'t':>2} {'ms':>8} {'GFLOP/s':>8} {'S_n':>6} {'Ptot':>6} "
          f"{'Pdyn':>6} {'Etot':>6} {'Edyn':>6}")
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[r["size"]][r["threads"]].append(r)
    for size in sorted(by):
        d = by[size]
        if 1 not in d:
            continue
        t1 = med([r["ms_median"] for r in d[1]])
        p1 = med([r["P_mean_W"] for r in d[1]])
        for th in sorted(d):
            tn = med([r["ms_median"] for r in d[th]])
            pn = med([r["P_mean_W"] for r in d[th]])
            s = t1 / tn
            ptot = pn / p1
            etot = ptot / s
            pdyn = (pn - pi) / (p1 - pi) if pi and p1 > pi else float("nan")
            print(f"{size:>6} {th:>2} {tn:>8.3f} "
                  f"{med([r['gflops'] for r in d[th]]):>8.1f} {s:>6.2f} {ptot:>6.2f} "
                  f"{pdyn:>6.2f} {etot:>6.3f} {pdyn/s if pdyn == pdyn else float('nan'):>6.3f}")


def llm_table(D, p_idle):
    recs = [r for r in load(os.path.join(D, "E_llm.jsonl")) if r.get("tokens_per_s")]
    if not recs:
        return
    pi = p_idle.get(2400000)
    print("\n=== E6  LLM decode, llama.cpp Qwen-0.5B Q4_K_M (2400 MHz) ===")
    print(f"{'t':>2} {'tok/s':>8} {'S_n':>6} {'P W':>6} {'Ptot':>6} {'Pdyn':>6} "
          f"{'Etot':>6} {'Edyn':>6} {'mJ/tok':>8}")
    by = defaultdict(list)
    for r in recs:
        by[r["threads"]].append(r)
    if 1 not in by:
        return
    s1 = med([r["tokens_per_s"] for r in by[1]])
    p1 = med([r["P_mean_W"] for r in by[1]])
    for th in sorted(by):
        sn = med([r["tokens_per_s"] for r in by[th]])
        pn = med([r["P_mean_W"] for r in by[th]])
        s = sn / s1
        ptot = pn / p1
        pdyn = (pn - pi) / (p1 - pi) if pi and p1 > pi else float("nan")
        print(f"{th:>2} {sn:>8.2f} {s:>6.2f} {pn:>6.2f} {ptot:>6.2f} {pdyn:>6.2f} "
              f"{ptot/s:>6.3f} {pdyn/s:>6.3f} {1000*pn/sn:>8.1f}")


def cotenant_table(D):
    recs = load(os.path.join(D, "E_cotenant.jsonl"))
    if not recs:
        return
    print("\n=== E7  the co-tenancy claim, tested ===")
    by = defaultdict(list)
    for r in recs:
        by[r["cond"]].append(r)
    for cond in ("infer4_alone", "infer1_alone", "infer1_plus_3cotenant",
                 "cotenant4_alone"):
        rs = by.get(cond, [])
        if not rs:
            continue
        lat = med([r.get("lat_median_ms") for r in rs if r.get("lat_median_ms")])
        p = med([r.get("P_mean_W") for r in rs if r.get("P_mean_W")])
        bogo = med([r.get("cotenant_bogo_ops") for r in rs
                    if r.get("cotenant_bogo_ops")])
        ips = med([r.get("throughput_ips") for r in rs if r.get("throughput_ips")])
        print(f"  {cond:>22}  lat={lat:8.2f} ms  ips={ips if ips == ips else 0:7.1f}  "
              f"P={p:5.2f} W  co-tenant bogo-ops={bogo if bogo == bogo else 0:.0f}")


def sustained_table(D):
    recs = load(os.path.join(D, "E_sustained.jsonl"))
    if not recs:
        return
    print("\n=== E8  five-minute sustained runs against the 6 s protocol ===")
    for r in sorted(recs, key=lambda r: (r["model"], r["threads"])):
        print(f"  {SHORT.get(r['model'], r['model']):>20} t={r['threads']} "
              f"{r.get('lat_median_ms', -1):8.2f} ms  p95={r.get('lat_p95_ms', -1):8.2f}  "
              f"P={r.get('P_mean_W', -1):5.2f} W  temp {r.get('temp_before_c', -1):.1f}"
              f"->{r.get('temp_after_c', -1):.1f} C  n={r.get('n_inferences', 0)}")


def governor_table(D):
    recs = [r for r in load(os.path.join(D, "E_governor.jsonl")) if "lat_median_ms" in r]
    if not recs:
        return
    print("\n=== E9  governor interaction (free-running clock) ===")
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[(r["model"], r["gov"])][r["threads"]].append(r)
    print(f"{'model':>20} {'governor':>12} {'t':>2} {'ms':>8} {'S_n':>6} {'P W':>6} "
          f"{'Etot':>6} {'f MHz':>7}")
    for key in sorted(by, key=lambda k: (k[0], k[1])):
        d = by[key]
        if 1 not in d:
            continue
        t1 = med([r["lat_median_ms"] for r in d[1]])
        p1 = med([r["P_mean_W"] for r in d[1]])
        for th in sorted(d):
            tn = med([r["lat_median_ms"] for r in d[th]])
            pn = med([r["P_mean_W"] for r in d[th]])
            s = t1 / tn
            f = med([r.get("freq_end_khz", 0) for r in d[th]]) / 1000.0
            print(f"{SHORT.get(key[0], key[0]):>20} {key[1]:>12} {th:>2} {tn:>8.2f} "
                  f"{s:>6.2f} {pn:>6.2f} {(pn/p1)/s:>6.3f} {f:>7.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    args = ap.parse_args()
    D = args.data
    p_idle = idle_table(D)
    thread_tables(D, p_idle)
    perf_table(D)
    ops_table(D)
    gemm_table(D, p_idle)
    llm_table(D, p_idle)
    cotenant_table(D)
    sustained_table(D)
    governor_table(D)


if __name__ == "__main__":
    main()
