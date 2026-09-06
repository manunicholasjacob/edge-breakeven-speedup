#!/usr/bin/env python3
"""Generate paper-v2/numbers_v2.tex from the revision campaign data.

Every number in the revised manuscript flows through here, so rerunning the
analysis regenerates the manuscript's numbers with no hand transcription.
Missing stages emit \\texttt{TBD} rather than failing, so the paper compiles
while the campaign is still running.
"""

import io
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D2 = os.path.join(ROOT, "data", "v2")
D1 = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "paper-v2", "numbers_v2.tex")

CNN8 = ["googlenet-12", "resnet18-v1-7", "efficientnet-lite4-11",
        "squeezenet1.1-7", "shufflenet-v2-10", "resnet50-v1-7",
        "mobilenetv2-12", "densenet-12"]
SHORT = {"googlenet-12": "GoogLeNet", "resnet18-v1-7": "ResNet-18",
         "efficientnet-lite4-11": "EfficientNet-Lite4",
         "squeezenet1.1-7": "SqueezeNet 1.1", "shufflenet-v2-10": "ShuffleNet-v2",
         "resnet50-v1-7": "ResNet-50", "mobilenetv2-12": "MobileNetV2",
         "densenet-12": "DenseNet-121", "mnv3-cifar": "MobileNetV3 (CIFAR)"}
M = {}


def load(path):
    out = []
    if os.path.exists(path):
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
    return st.median(xs) if xs else None


def put(k, v):
    M[k] = v if v is not None else r"\texttt{TBD}"


# ---------------------------------------------------------------- idle -----
_idle_all = [r for r in load(os.path.join(D2, "E_idle.jsonl"))
             if "P_mean_W" in r]
idle = [r for r in _idle_all if r.get("mode") in ("idle", "idle_after_load")]
_cold = [r for r in _idle_all if r.get("mode") == "idle"]
_warm = [r for r in _idle_all if r.get("mode") == "idle_after_load"]
P0 = {}
for r in idle:
    P0.setdefault(r["freq_khz"], []).append(r["P_mean_W"])
P0 = {k: med(v) for k, v in P0.items()}
P0core = {}
for r in idle:
    P0core.setdefault(r["freq_khz"], []).append(r.get("P_core_mean_W"))
P0core = {k: med(v) for k, v in P0core.items()}

put("npIdleFifteen", f"{P0[1500000]:.2f}" if 1500000 in P0 else None)
put("npIdleTwentyFour", f"{P0[2400000]:.2f}" if 2400000 in P0 else None)
if 2400000 in P0:
    put("npIdleErrTwentyFour", f"{100*abs(2.063-P0[2400000])/P0[2400000]:.0f}")
if 1500000 in P0:
    put("npIdleErrFifteen", f"{100*abs(2.063-P0[1500000])/P0[1500000]:.0f}")

idle_rows = []
for khz in sorted(P0):
    idle_rows.append((khz, P0[khz]))

put("npIdleEighteen", f"{P0[1800000]:.2f}" if 1800000 in P0 else None)
put("npIdleTwentyOne", f"{P0[2100000]:.2f}" if 2100000 in P0 else None)
# how many windows the floor rests on, and whether the two conditions differ
_per = defaultdict(int)
for r in idle:
    _per[r["freq_khz"]] += 1
put("npIdleNWindows", str(min(_per.values())) if _per else None)
_cw = {}
for tag, rows in (("Cold", _cold), ("Warm", _warm)):
    d = defaultdict(list)
    for r in rows:
        d[r["freq_khz"]].append(r["P_mean_W"])
    _cw[tag] = {k: med(v) for k, v in d.items()}
if _cw["Cold"] and _cw["Warm"]:
    _delta = [abs(_cw["Cold"][k] - _cw["Warm"][k])
              for k in _cw["Cold"] if k in _cw["Warm"]]
    put("npIdleColdWarmMax", f"{max(_delta):.2f}" if _delta else None)
    put("npIdleColdFifteen", f"{_cw['Cold'].get(1500000, 0):.2f}")
    put("npIdleWarmFifteen", f"{_cw['Warm'].get(1500000, 0):.2f}")
_mono = all(P0[a] < P0[b] for a, b in
            zip(sorted(P0), sorted(P0)[1:])) if len(P0) > 1 else False
put("npIdleMonotone", "rises monotonically with clock" if _mono
    else "is not monotone in clock")
# the widest spread among the windows at any one clock, which bounds how
# finely the floor, and therefore the dynamic threshold, can be read
_wid = {}
for r in idle:
    _wid.setdefault(r["freq_khz"], []).append(r["P_mean_W"])
_widp = [100 * (max(v) - min(v)) / min(v) for v in _wid.values() if len(v) > 1]
put("npIdleSpreadMaxPct", f"{max(_widp):.0f}" if _widp else None)
# spread of the idle measurement at each clock, which the floor claims need
_sp = {}
for r in idle:
    _sp.setdefault(r["freq_khz"], []).append(r["P_mean_W"])
put("npIdleSpreadEighteen",
    f"{max(_sp[1800000]) - min(_sp[1800000]):.2f}" if 1800000 in _sp else None)
put("npIdleSpreadTwentyFour",
    f"{max(_sp[2400000]) - min(_sp[2400000]):.3f}" if 2400000 in _sp else None)

# --------------------------------------------------------------- threads ---
th = [r for r in load(os.path.join(D2, "E_threads.jsonl"))
      if "lat_median_ms" in r and "P_mean_W" in r]
# Repetitions 4-6 re-measure the three-thread cells on a quiet board, after two
# contiguous blocks of the first pass were found to be contaminated. Both sets
# ship; the analysis uses all of them, and the contamination is characterised
# in its own right below.
th += [r for r in load(os.path.join(D2, "E_threads3.jsonl"))
       if "lat_median_ms" in r and "P_mean_W" in r]
by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in th:
    by[r["freq_khz"]][r["model"]][r["threads"]].append(r)


def cell(khz, model, thr, field):
    return med([r.get(field) for r in by[khz][model].get(thr, [])])


def by_rep(khz, model, thr, field):
    return {r["rep"]: r[field] for r in by[khz][model].get(thr, [])
            if field in r}


def ratios(khz, model, thr, rail="P_mean_W"):
    """Ratios formed within a repetition wherever both cells have it.

    The three-thread cells were re-measured, so they carry repetitions the
    one-thread reference does not. Intersecting the repetition sets, which is
    what an earlier version of this function did, silently discarded the clean
    re-measurements and left every three-thread ratio resting on the
    contaminated ones. Repetitions present only in the n-thread cell are
    therefore paired against the reference cell's median, which is the closest
    thing to matching that the design allows, and `n_matched` records how many
    ratios were genuinely matched.
    """
    t1r, tnr = by_rep(khz, model, 1, "lat_median_ms"), by_rep(khz, model, thr, "lat_median_ms")
    p1r, pnr = by_rep(khz, model, 1, rail), by_rep(khz, model, thr, rail)
    pi = P0.get(khz) if rail == "P_mean_W" else P0core.get(khz)
    ref = sorted(set(t1r) & set(p1r))
    reps = sorted(set(tnr) & set(pnr))
    if not (ref and reps):
        return None
    t1m, p1m = med([t1r[r] for r in ref]), med([p1r[r] for r in ref])
    S, PT, PD, ET, ED = [], [], [], [], []
    for r in reps:
        a = t1r[r] if r in t1r and r in p1r else t1m
        b = p1r[r] if r in t1r and r in p1r else p1m
        s = a / tnr[r]
        pt = pnr[r] / b
        S.append(s)
        PT.append(pt)
        ET.append(pt / s)
        if pi and b > pi:
            pd = (pnr[r] - pi) / (b - pi)
            PD.append(pd)
            ED.append(pd / s)
    return {"S": med(S), "Ptot": med(PT), "Pdyn": med(PD) if PD else None,
            "Etot": med(ET), "Edyn": med(ED) if ED else None,
            "t": cell(khz, model, thr, "lat_median_ms"),
            "P": cell(khz, model, thr, rail), "n_reps": len(reps),
            "n_matched": len(set(reps) & set(ref)),
            "S_spread": (max(S) - min(S)) / min(S) if len(S) > 1 else 0.0}


def board_threshold(khz, thr, dyn):
    """Median over the per-model ratios, not a pool over all repetitions."""
    vals = []
    for m in CNN8:
        r = ratios(khz, m, thr)
        if not r:
            continue
        v = r["Pdyn"] if dyn else r["Ptot"]
        if v:
            vals.append(v)
    return med(vals) if vals else None


for tag, khz in (("TwentyFour", 2400000), ("Fifteen", 1500000)):
    v = board_threshold(khz, 4, False)
    put("npBreakTotal" + tag, f"{v:.2f}" if v else None)
    v = board_threshold(khz, 4, True)
    put("npBreakDyn" + tag, f"{v:.2f}" if v else None)

K = 2400000
if by.get(K):
    rs = {m: ratios(K, m, 4) for m in CNN8 if ratios(K, m, 4)}
    if rs:
        put("npNModels", str(len(rs)))
        sv = [1 - r["Etot"] for r in rs.values() if r["Etot"] < 1]
        put("npSaveRangeTotal", f"{100*min(sv):.0f}--{100*max(sv):.0f}\\%" if sv else None)
        cd = [r["Edyn"] - 1 for r in rs.values() if r["Edyn"] and r["Edyn"] > 1]
        put("npCostRangeDyn", f"{100*min(cd):.0f}--{100*max(cd):.0f}\\%" if cd else None)
        flip = sum(1 for r in rs.values()
                   if r["Etot"] < 1 and r["Edyn"] and r["Edyn"] > 1)
        put("npNFlip", str(flip))
        s4 = [r["S"] for r in rs.values()]
        put("npSpeedRangeFour", f"{min(s4):.2f}--{max(s4):.2f}$\\times$")
        put("npSpeedMedFour", f"{st.median(s4):.2f}")
        s2 = [ratios(K, m, 2)["S"] for m in rs if ratios(K, m, 2)]
        put("npSpeedMedTwo", f"{st.median(s2):.2f}" if s2 else None)
        p4 = [r["Ptot"] for r in rs.values()]
        put("npPowRangeFour", f"{min(p4):.2f}--{max(p4):.2f}$\\times$")
        p2 = [ratios(K, m, 2)["Ptot"] for m in rs if ratios(K, m, 2)]
        put("npPowRangeTwo", f"{min(p2):.2f}--{max(p2):.2f}$\\times$" if p2 else None)
        t1s = [cell(K, m, 1, "lat_median_ms") for m in rs]
        if all(t1s):
            put("npWorkRange", f"{min(t1s):.1f}--{max(t1s):.0f}\\,ms")
        cv_by_t = {}
        for t in (1, 2, 3, 4):
            cvs = []
            for m in CNN8:
                lat = [r["lat_median_ms"] for r in by[K][m].get(t, [])]
                if len(lat) > 1:
                    cvs.append(100 * st.stdev(lat) / st.mean(lat))
            if cvs:
                cv_by_t[t] = (st.median(cvs), max(cvs))
        all_cv = []
        for t_ in (1, 2, 3, 4):
            for m in CNN8:
                lat = [r["lat_median_ms"] for r in by[K][m].get(t_, [])]
                if len(lat) > 1:
                    all_cv.append((t_, 100 * st.stdev(lat) / st.mean(lat)))
        if cv_by_t and all_cv:
            put("npCvOne", f"{cv_by_t[1][0]:.2f}" if 1 in cv_by_t else None)
            put("npCvFour", f"{cv_by_t[4][0]:.2f}" if 4 in cv_by_t else None)
            # the median over every cell, not over four per-thread medians
            put("npCvMedian", f"{st.median([v for _, v in all_cv]):.2f}")
            worst = max(all_cv, key=lambda kv: kv[1])
            put("npCvMax", f"{worst[1]:.0f}")
            put("npCvMaxThreads", str(worst[0]))
            put("npCvThreeMed", f"{cv_by_t[3][0]:.1f}" if 3 in cv_by_t else None)
            others = [v for k, v in all_cv if k != 3]
            put("npCvNonThreeMax", f"{max(others):.1f}" if others else None)
        pi = P0.get(K)
        p1m = med([cell(K, m, 1, "P_mean_W") for m in rs])
        p4m = med([cell(K, m, 4, "P_mean_W") for m in rs])
        if pi and p1m and p4m:
            put("npFloorShareOne", f"{100*pi/p1m:.0f}")
            put("npFloorShareFour", f"{100*pi/p4m:.0f}")
    mn = ratios(K, "mnv3-cifar", 4)
    if mn:
        put("npSpeedMnvFour", f"{mn['S']:.2f}$\\times$")
        put("npCostMnvTotal", f"{100*(mn['Etot']-1):.0f}\\%")
        put("npCostMnvDyn", f"{100*(mn['Edyn']-1):.0f}\\%" if mn["Edyn"] else None)

# dynamic energy is monotone in thread count: the sharpest form of the result
if by.get(K):
    med_dyn = {}
    best = None
    for thr in (2, 3, 4):
        vals = [ratios(K, m, thr)["Edyn"] for m in CNN8
                if ratios(K, m, thr) and ratios(K, m, thr)["Edyn"]]
        if vals:
            med_dyn[thr] = st.median(vals)
            lo = min(vals)
            if best is None or lo < best[0]:
                best = (lo, thr)
    if 2 in med_dyn:
        put("npEdynMedTwo", f"{med_dyn[2]:.2f}")
    if 4 in med_dyn:
        put("npEdynMedFour", f"{med_dyn[4]:.2f}")
    # the board-rail total-energy median, for comparison against the core rail
    _et = [ratios(K, m, 4)["Etot"] for m in CNN8
           if ratios(K, m, 4) and ratios(K, m, 4)["Etot"]]
    if _et:
        put("npEtotMedFour", f"{st.median(_et):.2f}")
    # networks for which the fourth core is a net loss in latency as well
    _w = 0
    for m in CNN8:
        a, b = ratios(K, m, 3), ratios(K, m, 4)
        if a and b and a["S"] >= b["S"]:
            _w += 1
    put("npNFourthWorse", {0:"none",1:"one",2:"two",3:"three",4:"four"}.get(_w, str(_w)))
    if best:
        put("npEdynBest", f"{best[0]:.3f}")
        put("npEdynBestThreads", str(best[1]))
    # how many cells anywhere in the sweep fall below break-even, and by how
    # much: one does, by less than the dispersion of the measurement
    under = []
    total = 0
    for khz in by:
        for m in CNN8 + ["mnv3-cifar"]:
            for thr in (2, 3, 4):
                r = ratios(khz, m, thr)
                if r and r["Edyn"]:
                    total += 1
                    if r["Edyn"] < 1:
                        under.append((r["Edyn"], m, khz, thr))
    put("npDynUnderCount", str(len(under)))
    put("npDynCellCount", str(total))
    if under:
        u = min(under)
        put("npDynUnderModel", SHORT.get(u[1], u[1]))
        put("npDynUnderThreads", str(u[3]))
        put("npDynUnderMargin", f"{100*(1-u[0]):.1f}")

# the floor a deployed board actually idles at, and what it does to the bar.
# Every comparison below is per model: a model's own P_1 and P_n decide its own
# energy, and the models with the highest speedups also carry the highest power
# ratios, so a board-median threshold would flatter them.
gov = [r for r in load(os.path.join(D2, "E_idlegov.jsonl")) if "P_mean_W" in r]
if gov and by.get(K):
    gb2 = defaultdict(list)
    for r in gov:
        gb2[r["mode"]].append(r["P_mean_W"])
    gfloor = med(gb2.get("schedutil", []) + gb2.get("ondemand", []))
    put("npFloorGov", f"{gfloor:.2f}" if gfloor else None)
    if gfloor and by.get(K):
        _bg = []
        for m in CNN8:
            p1 = cell(K, m, 1, "P_mean_W")
            p4 = cell(K, m, 4, "P_mean_W")
            if p1 and p4 and p1 > gfloor:
                _bg.append((p4 - gfloor) / (p1 - gfloor))
        put("npBreakDynGovBoard", f"{med(_bg):.2f}" if _bg else None)

    def per_model(floor):
        out = {}
        for m in CNN8:
            t1 = cell(K, m, 1, "lat_median_ms")
            t4 = cell(K, m, 4, "lat_median_ms")
            p1 = cell(K, m, 1, "P_mean_W")
            p4 = cell(K, m, 4, "P_mean_W")
            if None in (t1, t4, p1, p4) or p1 <= floor:
                continue
            s = t1 / t4
            ratio = (p4 - floor) / (p1 - floor)
            out[m] = {"S": s, "ratio": ratio, "E": ratio / s}
        return out

    pin = per_model(P0[K]) if K in P0 else {}
    gv = per_model(gfloor) if gfloor else {}
    if pin and gv:
        put("npNClearPinned", str(sum(1 for v in pin.values() if v["E"] < 1)))
        put("npNClearGov", str(sum(1 for v in gv.values() if v["E"] < 1)))
        put("npDynRatioRangePinned",
            f"{min(v['ratio'] for v in pin.values()):.2f}--"
            f"{max(v['ratio'] for v in pin.values()):.2f}")
        put("npDynRatioRangeGov",
            f"{min(v['ratio'] for v in gv.values()):.2f}--"
            f"{max(v['ratio'] for v in gv.values()):.2f}")
        put("npCostRangeDynGov",
            f"{100*(min(v['E'] for v in gv.values())-1):.0f}--"
            f"{100*(max(v['E'] for v in gv.values())-1):.0f}\\%")
        # the network that comes closest, under the more favourable floor
        cm = min(gv, key=lambda m: gv[m]["E"])
        put("npClosestModel", SHORT.get(cm, cm))
        put("npClosestSpeed", f"{gv[cm]['S']:.2f}")
        put("npClosestRatio", f"{gv[cm]['ratio']:.2f}")
        put("npClosestShortPct", f"{100*(gv[cm]['E']-1):.0f}")
        # total-accounting power-ratio spread, for the calibration caveat
        tot = [cell(K, m, 4, "P_mean_W") / cell(K, m, 1, "P_mean_W")
               for m in CNN8 if cell(K, m, 1, "P_mean_W")]
        put("npTotRatioRange", f"{min(tot):.2f}--{max(tot):.2f}")

# the ImageNet-scale network whose service time sits inside the batch sweep's
# range, quoted in the mechanism section
sh = ratios(K, "shufflenet-v2-10", 4)
if sh:
    put("npShuffleSFour", f"{sh['S']:.2f}")
    put("npShuffleTOne", f"{cell(K, 'shufflenet-v2-10', 1, 'lat_median_ms'):.1f}")
# the dynamic share of the signal, which is the complement of the floor share
if by.get(K) and K in P0:
    _p1 = med([cell(K, m, 1, "P_mean_W") for m in CNN8])
    _p4 = med([cell(K, m, 4, "P_mean_W") for m in CNN8])
    if _p1 and _p4:
        put("npDynShareOne", f"{100*(1 - P0[K]/_p1):.0f}")
        put("npDynShareFour", f"{100*(1 - P0[K]/_p4):.0f}")
# core-rail robustness: the same comparison with board peripherals removed
if by.get(K) and K in P0core:
    core = [ratios(K, m, 4, rail="P_core_mean_W") for m in CNN8]
    core = [c for c in core if c and c["Edyn"]]
    if core:
        put("npCoreRailEdynMed", f"{st.median([c['Edyn'] for c in core]):.2f}")
        put("npCoreRailEdynRange",
            f"{min(c['Edyn'] for c in core):.2f}--{max(c['Edyn'] for c in core):.2f}")
        put("npCoreRailNSave", str(sum(1 for c in core if c["Edyn"] < 1)))
        put("npCoreRailEtotMed", f"{st.median([c['Etot'] for c in core]):.2f}")
        put("npCoreRailNSaveTot", str(sum(1 for c in core if c["Etot"] < 1)))
        put("npCoreFloor", f"{P0core[K]:.2f}")
        put("npNonCoreFloor", f"{P0[K] - P0core[K]:.2f}")

# the spin-wait test: it refutes the hypothesis it was built to check
spin = [r for r in load(os.path.join(D2, "E_spin.jsonl")) if "lat_median_ms" in r]
if spin:
    sb = defaultdict(lambda: defaultdict(list))
    for r in spin:
        if r["threads"] == 3:
            sb[r["spin"]][r["model"]].append(r["lat_median_ms"])
    sp = {}
    for k in ("1", "0"):
        v = [100 * (max(x) - min(x)) / min(x) for x in sb[k].values() if len(x) > 1]
        if v:
            sp[k] = st.median(v)
    if len(sp) == 2:
        put("npSpinOnSpread", f"{sp['1']:.0f}")
        put("npSpinOffSpread", f"{sp['0']:.0f}")

# what the instrument actually delivered, rather than what it was designed for
if th:
    rates = [r["n_pmic_samples"] / r["pmic_span_s"] for r in th
             if r.get("n_pmic_samples") and r.get("pmic_span_s")]
    if rates:
        put("npSampleRate", f"{st.median(rates):.0f}")
    for thr, name in ((1, "npSamplesOne"), (4, "npSamplesFour")):
        n = med([r["n_pmic_samples"] for r in th
                 if r.get("threads") == thr and r.get("wall_s", 0) < 10
                 and r.get("n_pmic_samples")])
        put(name, f"{n:.0f}" if n else None)

# tables ---------------------------------------------------------------------
rows = []
for khz, P0v in idle_rows:
    # Thresholds are reported only at 2400 MHz. The pure-load control shows the
    # summed PMIC rails do not report board power faithfully at 1500 MHz under
    # multi-core load, so any threshold derived from power there would be
    # unsound; the floor itself is a single-core-idle measurement and stands.
    if khz == K:
        tot2 = board_threshold(khz, 2, False)
        dyn2 = board_threshold(khz, 2, True)
        tot4 = board_threshold(khz, 4, False)
        dyn4 = board_threshold(khz, 4, True)
    else:
        tot2 = dyn2 = tot4 = dyn4 = None
    fmt = lambda x: f"{x:.2f}" if x else "--"
    rows.append(f"{khz//1000} & {P0v:.2f} & {fmt(tot2)} & {fmt(dyn2)} & "
                f"{fmt(tot4)} & {fmt(dyn4)} \\\\")
M["npTableIdle"] = "\n".join(rows) if rows else "-- & -- & -- & -- & -- & -- \\\\"

rows = []
if by.get(K):
    for m in CNN8 + ["mnv3-cifar"]:
        if m not in by[K]:
            continue
        r1 = cell(K, m, 1, "lat_median_ms")
        cells = []
        for t in (2, 3, 4):
            r = ratios(K, m, t)
            cells.append(f"{r['S']:.2f}" if r else "--")
        r4 = ratios(K, m, 4)
        if not r4:
            continue
        bold = lambda v: (f"\\textbf{{{v:.2f}}}" if v < 1 else f"{v:.2f}")
        rows.append(f"{SHORT.get(m, m)} & {r1:.1f} & " + " & ".join(cells) +
                    f" & {r4['Ptot']:.2f} & {bold(r4['Etot'])} & "
                    f"{bold(r4['Edyn'])} \\\\")
M["npTableThreads"] = "\n".join(rows) if rows else "-- \\\\"

# ------------------------------------------------------------------ perf ---
# E_perf2 counts the inference loop by subtracting a startup-only run and asks
# for no more events than the PMU can count at once; E_perf, which counted the
# whole process, is superseded and kept only in the artifact.
perf = [r for r in load(os.path.join(D2, "E_perf2.jsonl")) if r.get("counters")]
if not perf:
    perf = [r for r in load(os.path.join(D2, "E_perf.jsonl")) if r.get("counters")]
pb = defaultdict(lambda: defaultdict(list))
for r in perf:
    pb[(r["model"], r.get("res"))][r["threads"]].append(r)


def pc(key, thr, name):
    rs = pb[key].get(thr, [])
    return med([r["counters"].get(name) for r in rs])


def per_inf(key, thr, name):
    rs = pb[key].get(thr, [])
    v = med([r["counters"].get(name) for r in rs])
    n = med([r["run"].get("n_iter") for r in rs])
    return (v / n) if (v and n) else None


# the paper shows the three ResNet-18 resolutions, which isolate operator
# extent with the graph held fixed, plus the CIFAR-scale network; the remaining
# targets are in the artifact
SHOW = [("resnet18-v1-7", 224), ("resnet18-v1-7", 96), ("resnet18-v1-7", 32),
        ("mnv3-cifar", None)]
rows = []
for key in [k for k in SHOW if k in pb]:
    for thr in sorted(pb[key]):
        cyc = pc(key, thr, "cycles")
        ins = pc(key, thr, "instructions")
        stall = pc(key, thr, "stalled-cycles-backend")
        l2 = per_inf(key, thr, "l2d_cache_refill")
        bus = per_inf(key, thr, "bus_access")
        if not (cyc and ins):
            continue
        label = SHORT.get(key[0], key[0])
        if key[1]:
            label += f" @{key[1]}"
        n_i = per_inf(key, thr, "instructions")
        rows.append(f"{label} & {thr} & {n_i/1e6:.0f} & {ins/cyc:.2f} & "
                    f"{100*stall/cyc:.0f} & {l2/1000:.0f}k \\\\")
M["npTablePerf"] = "\n".join(rows) if rows else "-- \\\\"

if perf and "implied_busy_cores" in perf[0]:
    put("npMaxBusyCores", f"{max(r['implied_busy_cores'] for r in perf):.2f}")
    _run = [min(r["pct_running"].values()) for r in perf if r.get("pct_running")]
    put("npMinCounterRunning", f"{min(_run):.0f}" if _run else None)

key224 = ("resnet18-v1-7", 224)
if pb.get(key224):
    l2_1, l2_4 = per_inf(key224, 1, "l2d_cache_refill"), per_inf(key224, 4, "l2d_cache_refill")
    bus1, bus4 = per_inf(key224, 1, "bus_access"), per_inf(key224, 4, "bus_access")
    c1, c4 = pc(key224, 1, "cycles"), pc(key224, 4, "cycles")
    i1, i4 = pc(key224, 1, "instructions"), pc(key224, 4, "instructions")
    s1, s4c = pc(key224, 1, "stalled-cycles-backend"), pc(key224, 4, "stalled-cycles-backend")
    if l2_1 and l2_4:
        put("npLTwoGrowthPct", f"{100*(l2_4/l2_1-1):.0f}")
    if bus1 and bus4:
        put("npBusGrowthPct", f"{100*(bus4/bus1-1):.0f}")
    if c1 and i1:
        put("npIpcOne", f"{i1/c1:.2f}")
        put("npStallOne", f"{100*s1/c1:.0f}")
    if c4 and i4:
        put("npIpcFour", f"{i4/c4:.2f}")
        put("npStallFour", f"{100*s4c/c4:.0f}")

# instruction growth: the runtime's own overhead, per inference
def _instr_per_inf(key, thr):
    rs = pb[key].get(thr, [])
    v = med([r["counters"].get("instructions") for r in rs])
    n = med([r["run"].get("n_iter") for r in rs])
    return (v / n) if (v and n) else None


for key, tag in ((("resnet18-v1-7", 224), "Big"), (("resnet18-v1-7", 32), "Small")):
    a, b = _instr_per_inf(key, 1), _instr_per_inf(key, 4)
    if a and b:
        put("npInstrGrowth" + tag, f"{100*(b/a-1):.0f}")
key32 = ("resnet18-v1-7", 32)
if pb.get(key32):
    c1, i1 = pc(key32, 1, "cycles"), pc(key32, 1, "instructions")
    c4, i4 = pc(key32, 4, "cycles"), pc(key32, 4, "instructions")
    if c4 and i4:
        put("npIpcSmallFour", f"{i4/c4:.2f}")
    if c1 and i1:
        put("npIpcSmallOne", f"{i1/c1:.2f}")

# MobileNetV3 at 32x32 is the case that inverts the memory reading: its added
# threads retire far more instructions at a HIGHER rate and stall LESS.
keymnv = ("mnv3-cifar", None)
if pb.get(keymnv):
    a, b = _instr_per_inf(keymnv, 1), _instr_per_inf(keymnv, 4)
    if a and b:
        put("npInstrGrowthMnv", f"{100*(b/a-1):.0f}")
    c1, i1 = pc(keymnv, 1, "cycles"), pc(keymnv, 1, "instructions")
    c4, i4 = pc(keymnv, 4, "cycles"), pc(keymnv, 4, "instructions")
    s1 = pc(keymnv, 1, "stalled-cycles-backend")
    s4 = pc(keymnv, 4, "stalled-cycles-backend")
    if c1 and i1:
        put("npIpcMnvOne", f"{i1/c1:.2f}")
        put("npStallMnvOne", f"{100*s1/c1:.0f}")
    if c4 and i4:
        put("npIpcMnvFour", f"{i4/c4:.2f}")
        put("npStallMnvFour", f"{100*s4/c4:.0f}")

# bus accesses grow less in relative terms but more in absolute terms, which is
# why the counters cannot localise the contention on their own
if pb.get(key224):
    _l2a = per_inf(key224, 4, "l2d_cache_refill") - per_inf(key224, 1, "l2d_cache_refill")
    _bua = per_inf(key224, 4, "bus_access") - per_inf(key224, 1, "bus_access")
    if _l2a and _bua:
        put("npLTwoGrowthAbs", f"{_l2a/1e3:.0f}")
        put("npBusGrowthAbs", f"{_bua/1e6:.1f}")
        put("npBusOverLTwo", f"{_bua/_l2a:.0f}")
    _b1 = per_inf(key224, 1, "bus_access")
    _l1 = per_inf(key224, 1, "l2d_cache_refill")
    if _b1 and _l1:
        put("npBusPerLTwoBaseline", f"{_b1/_l1:.0f}")

# spin-wait test and three-thread repetition count
spin = [r for r in load(os.path.join(D2, "E_spin.jsonl")) if "lat_median_ms" in r]
if spin:
    sb = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in spin:
        sb[r["model"]][r["threads"]][r["spin"]].append(r)
    spreads = {}
    for m in sb:
        for thr in sb[m]:
            for sp in ("0", "1"):
                lats = [r["lat_median_ms"] for r in sb[m][thr].get(sp, [])]
                if len(lats) > 1:
                    spreads[(m, thr, sp)] = 100 * (max(lats) - min(lats)) / min(lats)
    on3 = [v for (m, thr, sp), v in spreads.items() if thr == 3 and sp == "1"]
    off3 = [v for (m, thr, sp), v in spreads.items() if thr == 3 and sp == "0"]

n3 = len({(r["model"], r["rep"]) for r in th
          if r.get("threads") == 3 and r.get("freq_khz") == 2400000})
n3models = len({r["model"] for r in th if r.get("threads") == 3})
put("npThreeReps", str(n3 // n3models) if n3models else None)

# ------------------------------------------------------------------- ops ---
ops = load(os.path.join(D2, "E_ops3.jsonl")) or load(os.path.join(D2, "E_ops.jsonl"))
# E_ops3 carries three repetitions per cell. A dict comprehension keyed on
# (res, threads) silently keeps only the last one, which is the same defect the
# ratio function had. Reduce each operator's timing by the median instead.
_acc = defaultdict(lambda: defaultdict(list))
for r in ops:
    for _opname, _d in (r.get("ops") or {}).items():
        _acc[(r["res"], r["threads"])][_opname].append(_d)
obr = {}
for _key, _opmap in _acc.items():
    obr[_key] = {"res": _key[0], "threads": _key[1],
                 "ops": {k: {"n": med([d["n"] for d in v]),
                             "us": med([d["us"] for d in v])}
                         for k, v in _opmap.items()}}
rows = []
for res in sorted({r["res"] for r in ops}):
    a, b = obr.get((res, 1)), obr.get((res, 4))
    if not (a and b):
        continue
    tot1 = sum(v["us"] for v in a["ops"].values())
    tot4 = sum(v["us"] for v in b["ops"].values())
    # the runtime reports fused and unfused convolutions separately; both are
    # convolution work and both are what the thread pool divides
    CONV = ("Conv", "FusedConv")
    conv1 = sum(a["ops"].get(k, {}).get("us", 0) for k in CONV)
    conv4 = sum(b["ops"].get(k, {}).get("us", 0) for k in CONV)
    rows.append(f"{res}$\\times${res} & {tot1/1000:.1f} & {tot1/tot4:.2f} & "
                f"{100*conv1/tot1:.0f} & {conv1/conv4 if conv4 else 0:.2f} \\\\")
    if res == 224:
        put("npConvShareBig", f"{100*conv1/tot1:.0f}")
        put("npConvSpeedBig", f"{conv1/conv4:.2f}" if conv4 else None)
    if res == 32:
        put("npConvShareSmall", f"{100*conv1/tot1:.0f}")
        put("npConvSpeedSmall", f"{conv1/conv4:.2f}" if conv4 else None)
M["npTableOps"] = "\n".join(rows) if rows else "-- \\\\"

# ------------------------------------------------------------ gemm / llm ---
# rep-matched, like the CNN sweep: each ratio is formed inside a repetition
gm = [r for r in load(os.path.join(D2, "E_gemm.jsonl")) if "ms_median" in r]
gb = defaultdict(lambda: defaultdict(list))
for r in gm:
    gb[r["size"]][r["threads"]].append(r)
sp, best = [], None
pi24 = P0.get(2400000)
for size in sorted(gb):
    t1 = med([r["ms_median"] for r in gb[size].get(1, [])])
    p1 = med([r["P_mean_W"] for r in gb[size].get(1, [])])
    for thr in (2, 3, 4):
        tn = med([r["ms_median"] for r in gb[size].get(thr, [])])
        pn = med([r["P_mean_W"] for r in gb[size].get(thr, [])])
        if not (t1 and tn and p1 and pn and pi24):
            continue
        s = t1 / tn
        edyn = ((pn - pi24) / (p1 - pi24)) / s
        if thr == 4:
            sp.append(s)
        if best is None or edyn < best[0]:
            best = (edyn, size, thr, s)
put("npGemmSpeedRange", f"{min(sp):.2f}--{max(sp):.2f}$\\times$" if sp else None)
if best:
    put("npGemmBestEdyn", f"{best[0]:.2f}")
    put("npGemmBestSize", str(best[1]))
    put("npGemmBestThreads", str(best[2]))
    put("npGemmBestSpeed", f"{best[3]:.2f}")
    put("npGemmBestWorkingSetMB", f"{3 * best[1] ** 2 * 4 / 1048576:.1f}")
# every configuration that clears the dynamic bar, and the fastest one measured
clears, fastest = [], None
for size in sorted(gb):
    t1 = med([r["ms_median"] for r in gb[size].get(1, [])])
    p1 = med([r["P_mean_W"] for r in gb[size].get(1, [])])
    for thr in (2, 3, 4):
        tn = med([r["ms_median"] for r in gb[size].get(thr, [])])
        pn = med([r["P_mean_W"] for r in gb[size].get(thr, [])])
        if not (t1 and tn and p1 and pn and pi24):
            continue
        s = t1 / tn
        edyn = ((pn - pi24) / (p1 - pi24)) / s
        if edyn < 1.0:
            clears.append((size, thr, s, edyn))
        if fastest is None or s > fastest[2]:
            fastest = (size, thr, s, edyn)
if clears:
    put("npGemmNClears", str(len(clears)))
    put("npGemmClearList", ", ".join(
        f"$n={c[0]}$ at {c[1]} threads ({c[3]:.2f}$\\times$)" for c in clears))
if fastest:
    put("npGemmFastSize", str(fastest[0]))
    put("npGemmFastThreads", str(fastest[1]))
    put("npGemmFastSpeed", f"{fastest[2]:.2f}")
    put("npGemmFastEdyn", f"{fastest[3]:.2f}")
    put("npGemmFastWorkingSetKB", f"{3 * fastest[0] ** 2 * 4 / 1024:.0f}")

lm = [r for r in load(os.path.join(D2, "E_llm.jsonl")) if r.get("tokens_per_s")]
lb = defaultdict(list)
for r in lm:
    lb[r["threads"]].append(r)
if 1 in lb and 4 in lb:
    s1 = med([r["tokens_per_s"] for r in lb[1]])
    p1_llm = med([r["P_mean_W"] for r in lb[1]])
    s4v = med([r["tokens_per_s"] for r in lb[4]])
    e1 = med([1000 * r["P_mean_W"] / r["tokens_per_s"] for r in lb[1]])
    e4 = med([1000 * r["P_mean_W"] / r["tokens_per_s"] for r in lb[4]])
    put("npLlmSpeedFour", f"{s4v/s1:.2f}")
    put("npLlmEnergyPct", f"{100*(e4/e1-1):.0f}")
    # the LLM's best configuration under each accounting
    rows = {}
    for thr in sorted(lb):
        sn = med([r["tokens_per_s"] for r in lb[thr]])
        pn = med([r["P_mean_W"] for r in lb[thr]])
        if not (sn and pn):
            continue
        s = sn / s1
        rows[thr] = {"Etot": (pn / p1_llm) / s,
                     "Edyn": (((pn - pi24) / (p1_llm - pi24)) / s) if pi24 else None,
                     "mJ": 1000 * pn / sn}
    multi = {k: v for k, v in rows.items() if k >= 2}
    if multi:
        bt = min(multi, key=lambda k: multi[k]["Etot"])
        put("npLlmBestTotThreads", str(bt))
        put("npLlmBestTotSave", f"{100*(1-multi[bt]['Etot']):.0f}")
        bd = min(multi, key=lambda k: multi[k]["Edyn"] or 9)
        put("npLlmBestDyn", f"{multi[bd]['Edyn']:.2f}")
        put("npLlmBestDynThreads", str(bd))
        put("npLlmDynFour", f"{multi[4]['Edyn']:.2f}" if 4 in multi else None)

# Numbers carried over from the first campaign's controls, computed here so
# they cannot drift out of step with the data again.
def _v1(name):
    return load(os.path.join(D1, name))


batch = [r for r in _v1("P15b_batch_threshold.jsonl") if "ms_per_batch" in r]
if batch:
    bb = defaultdict(lambda: defaultdict(list))
    for r in batch:
        bb[r["batch"]][r["threads"]].append(r)
    s4s, e4s, t1s = [], [], []
    for bt in sorted(bb):
        d = bb[bt]
        if 1 not in d or 4 not in d:
            continue
        t1 = med([r["ms_per_batch"] for r in d[1]])
        t4 = med([r["ms_per_batch"] for r in d[4]])
        p1 = med([r["P_mean_W"] for r in d[1]])
        p4 = med([r["P_mean_W"] for r in d[4]])
        s = t1 / t4
        s4s.append(s)
        e4s.append((p4 / p1) / s)
        t1s.append(t1)
    if s4s:
        put("npBatchMaxSpeed", f"{max(s4s):.2f}")
        put("npBatchMinEnergy", f"{min(e4s):.2f}")
        put("npBatchWorkRange", f"{min(t1s):.2f}--{max(t1s):.1f}")
        put("npBatchFactor", f"{max(t1s)/min(t1s):.0f}")

resl = [r for r in _v1("P15c_resolution.jsonl") + _v1("P15d_resolution_more.jsonl")
        if "ms_per_inf" in r and r.get("resolution")]
if resl:
    rb = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in resl:
        rb[r["model"].replace(".onnx", "")][r["resolution"]][r["threads"]].append(r)
    crossings = {}
    for m in rb:
        prev = None
        for res in sorted(rb[m]):
            d = rb[m][res]
            if 1 not in d or 4 not in d:
                continue
            t1 = med([r["ms_per_inf"] for r in d[1]])
            t4 = med([r["ms_per_inf"] for r in d[4]])
            p1 = med([r["P_mean_W"] for r in d[1]])
            p4 = med([r["P_mean_W"] for r in d[4]])
            e = (p4 / p1) / (t1 / t4)
            if m == "resnet18-v1-7":
                if res == 32:
                    put("npResSpeedLow", f"{t1/t4:.2f}")
                if res == 224:
                    put("npResSpeedHigh", f"{t1/t4:.2f}")
            if prev and prev[1] > 1.0 >= e and m not in crossings:
                r0, e0 = prev
                crossings[m] = r0 + (res - r0) * (e0 - 1.0) / (e0 - e)
            prev = (res, e)
    if crossings:
        put("npCrossRange",
            f"{min(crossings.values()):.0f} and {max(crossings.values()):.0f}")

# ------------------------------------------------- sustained / gov / coten --
sus = load(os.path.join(D2, "E_sustained.jsonl"))
sus += load(os.path.join(D2, "E_sustained3.jsonl"))
if sus and by.get(K):
    d, dp = [], []
    for r in sus:
        m, t = r["model"], r["threads"]
        base = cell(K, m, t, "lat_median_ms")
        basep = cell(K, m, t, "P_mean_W")
        if base and r.get("lat_median_ms"):
            d.append(abs(r["lat_median_ms"] - base) / base)
        if basep and r.get("P_mean_W"):
            dp.append(abs(r["P_mean_W"] - basep) / basep)
    put("npSustainedDriftPct", f"{100*max(d):.0f}" if d else None)
    put("npSustainedPowerPct", f"{100*max(dp):.0f}" if dp else None)
    put("npSustainedMaxInf", f"{max(r.get('n_inferences', 0) for r in sus):,}"
        .replace(",", "{,}"))
    # the quantity the paper actually decides on is the energy ratio, so check
    # that rather than latency and power separately
    _sb = defaultdict(lambda: defaultdict(list))
    for r in sus:
        if r.get("lat_median_ms") and r.get("P_mean_W"):
            _sb[r["model"]][r["threads"]].append(r)
    _de = []
    for m, d4 in _sb.items():
        if 1 in d4 and 4 in d4:
            t1 = med([r["lat_median_ms"] for r in d4[1]])
            t4 = med([r["lat_median_ms"] for r in d4[4]])
            p1 = med([r["P_mean_W"] for r in d4[1]])
            p4 = med([r["P_mean_W"] for r in d4[4]])
            ref = ratios(K, m, 4)
            if ref and ref["Etot"]:
                _de.append(abs((p4 / p1) / (t1 / t4) - ref["Etot"]) / ref["Etot"])
    put("npSustainedEnergyPct", f"{100*max(_de):.1f}" if _de else None)

gov = [r for r in load(os.path.join(D2, "E_governor.jsonl")) if "lat_median_ms" in r]
if gov and by.get(K):
    deltas = []
    gg = defaultdict(lambda: defaultdict(list))
    for r in gov:
        gg[(r["model"], r["gov"])][r["threads"]].append(r)
    for (m, g), d in gg.items():
        if 1 in d and 4 in d:
            t1 = med([r["lat_median_ms"] for r in d[1]])
            t4 = med([r["lat_median_ms"] for r in d[4]])
            p1 = med([r["P_mean_W"] for r in d[1]])
            p4 = med([r["P_mean_W"] for r in d[4]])
            e = (p4 / p1) / (t1 / t4)
            ref = ratios(K, m, 4)
            if ref:
                deltas.append(abs(e - ref["Etot"]) / ref["Etot"])
    put("npGovernorDeltaPct", f"{100*max(deltas):.0f}" if deltas else None)

# The first co-tenancy run let the load generator run for 30 s against a 6 s
# measurement window and compared it to an 8 s reference, so its throughput
# share was an artifact of two hardcoded durations. E_cotenant2 gives every
# condition the same window and adds a three-core reference.
cot = load(os.path.join(D2, "E_cotenant2.jsonl"))
if not cot:
    cot = load(os.path.join(D2, "E_cotenant.jsonl"))
cb = defaultdict(list)
for r in cot:
    cb[r["cond"]].append(r)


def _c(cond, field):
    return med([r.get(field) for r in cb.get(cond, [])])


a = _c("infer1_alone", "lat_median_ms")
b = _c("infer1_plus_3cotenant", "lat_median_ms")
p1c = _c("infer1_plus_3cotenant", "P_mean_W")
p4c = _c("infer4_alone", "P_mean_W")
r_with = _c("infer1_plus_3cotenant", "cotenant_bogo_per_core_s")
r_alone = _c("cotenant3_alone", "cotenant_bogo_per_core_s")
if a and b:
    infl = 100 * (b / a - 1)
    put("npCotenantInflationPct", f"{infl:.1f}")
    share = 100 * r_with / r_alone if (r_with and r_alone) else None
    if share:
        put("npCotenantSharePct", f"{share:.0f}")
    M["npCotenantVerdict"] = (
        "The reasoning holds on this board. Single-threaded inference slows by "
        f"{infl:.1f}\\% when three co-tenant threads occupy the other cores "
        f"({a:.0f} to {b:.0f}\\,ms), and those three cores deliver "
        + (f"{share:.0f}\\% of the throughput per core that they reach with the "
           "board to themselves" if share else "co-tenant work throughout")
        + f", at {p1c:.2f}\\,W against {p4c:.2f}\\,W for four-thread inference "
        "alone. Each condition is measured over the same window, and the "
        "co-tenant reference runs on the same three cores rather than four.")
M.setdefault("npCotenantVerdict", r"\texttt{TBD}")


# ------------------------------------------------- the slow three-thread mode
# Incidence across every campaign that could show it, so the manuscript can
# state how often it occurs rather than describe one block of it.
def _incidence(rows, keyf, thresh=1.25):
    b = defaultdict(list)
    for r in rows:
        if "lat_median_ms" in r:
            b[keyf(r)].append(r)
    n_cell = n_hit = 0
    ratios, rel, maj = [], [], 0
    for k, v in b.items():
        if len(v) < 3:
            continue
        n_cell += 1
        m = st.median([x["lat_median_ms"] for x in v])
        hit = [x for x in v if x["lat_median_ms"] > thresh * m]
        ok = [x for x in v if x["lat_median_ms"] <= thresh * m]
        if hit:
            n_hit += 1
            ratios += [x["lat_median_ms"] / m for x in hit]
        if len(hit) * 2 > len(v):
            maj += 1
        # power and start temperature of each slow repetition against the
        # median of the other repetitions of the same cell
        if hit and ok:
            pm = med([x.get("P_mean_W") for x in ok])
            tm = med([x.get("temp_before_c") for x in ok])
            for x in hit:
                rel.append((x.get("P_mean_W"), pm,
                            x.get("temp_before_c"), tm))
    return n_cell, n_hit, ratios, rel, maj


_camps = [
    ("Sweep", [r for r in load(os.path.join(D2, "E_threads.jsonl"))],
     lambda r: (r["freq_khz"], r["model"], r["threads"])),
    ("Rerun", [r for r in load(os.path.join(D2, "E_threads3.jsonl"))],
     lambda r: (r["freq_khz"], r["model"], r["threads"])),
    ("Fifteen", [r for r in load(os.path.join(D2, "E_recheck15.jsonl"))],
     lambda r: (r["model"], r["threads"])),
    ("Spin", [r for r in load(os.path.join(D2, "E_spin.jsonl"))],
     lambda r: (r["model"], r["threads"], r["spin"])),
    ("Pinned", [r for r in load(os.path.join(D2, "E_isolate.jsonl"))],
     lambda r: (r["model"], r["threads"], r["pinned"])),
    ("Quiet", [r for r in load(os.path.join(D2, "E_bimodal.jsonl"))],
     lambda r: (r["model"], r["threads"])),
]
_allrel, _allrat, _maj_total = [], [], 0
for tag, rows, kf in _camps:
    n_cell, n_hit, _rats, rel, maj = _incidence(rows, kf)
    put("npBimodal" + tag + "Cells", str(n_cell))
    put("npBimodal" + tag + "Hits", str(n_hit))
    _allrel += rel
    _allrat += _rats
    _maj_total += maj
if _allrat:
    put("npBimodalRatioLo", f"{min(_allrat):.2f}")
    put("npBimodalRatioHi", f"{max(_allrat):.2f}")
    put("npBimodalRatioMed", f"{st.median(_allrat):.2f}")
    put("npBimodalMajority", str(_maj_total))
    _dp = [100 * (a / b - 1) for a, b, _, _ in _allrel if a and b]
    _dt = [a - b for _, _, a, b in _allrel if a is not None and b is not None]
    if _dp:
        put("npBimodalPowerDeltaPct", f"{abs(st.median(_dp)):.0f}")
        put("npBimodalPowerLower",
            str(sum(1 for x in _dp if x < 0)) + " of " + str(len(_dp)))
    if _dt:
        put("npBimodalTempDelta", f"{abs(st.median(_dt)):.1f}")

_slow_rows = []
for tag, rows, kf in _camps:
    b = defaultdict(list)
    for r in rows:
        if "lat_median_ms" in r:
            b[kf(r)].append(r)
    for k, v in b.items():
        if len(v) < 3:
            continue
        m = st.median([x["lat_median_ms"] for x in v])
        _slow_rows += [x for x in v if x["lat_median_ms"] > 1.25 * m]
if _slow_rows:
    _thr = sorted({r["threads"] for r in _slow_rows})
    put("npBimodalThreads", " and ".join(str(x) for x in _thr))
    _off = [r for r in _slow_rows
            if r.get("freq_actual_khz") not in (None, r.get("freq_khz", 2400000))]
    put("npBimodalOffClock", str(len(_off)))

# the dedicated incidence run: ten repetitions per thread count, nothing else
# queued on the board, which is the condition the main sweep does not meet
_bm = [r for r in load(os.path.join(D2, "E_bimodal.jsonl"))
       if "lat_median_ms" in r]
if _bm:
    _bb = defaultdict(list)
    for r in _bm:
        _bb[r["threads"]].append(r["lat_median_ms"])
    put("npBimodalQuietReps", str(min(len(v) for v in _bb.values())))
    _sp2 = [100 * (max(v) - min(v)) / min(v) for v in _bb.values() if len(v) > 1]
    put("npBimodalQuietSpread", f"{max(_sp2):.1f}" if _sp2 else None)
    put("npBimodalQuietTotal", str(len(_bm)))

# ------------------------------------------- the pinned-sampler control ------
_iso = [r for r in load(os.path.join(D2, "E_isolate.jsonl"))
        if "lat_median_ms" in r]
if _iso:
    _ib = defaultdict(list)
    for r in _iso:
        _ib[(r["threads"], str(r["pinned"]))].append(r)

    def _im(th_, pin, f):
        v = _ib.get((th_, pin), [])
        return med([r.get(f) for r in v])
    _res = {}
    for pin, tag in (("1", "Pinned"), ("0", "Free")):
        t1, t3 = _im(1, pin, "lat_median_ms"), _im(3, pin, "lat_median_ms")
        p1, p3 = _im(1, pin, "P_mean_W"), _im(3, pin, "P_mean_W")
        if t1 and t3 and p1 and p3:
            _res[tag] = (t1 / t3, p3 / p1)
            put("npIsoSpeed" + tag, f"{t1/t3:.2f}")
            put("npIsoPowRatio" + tag, f"{p3/p1:.2f}")
    if len(_res) == 2:
        put("npIsoSpeedDeltaPct",
            f"{100*abs(_res['Pinned'][0]-_res['Free'][0])/_res['Free'][0]:.1f}")
        put("npIsoPowDeltaPct",
            f"{100*abs(_res['Pinned'][1]-_res['Free'][1])/_res['Free'][1]:.1f}")
    _rates = [r["n_pmic_samples"] / r["pmic_span_s"] for r in _iso
              if r.get("n_pmic_samples") and r.get("pmic_span_s")]
    if _rates:
        put("npIsoSampleRate", f"{st.median(_rates):.0f}")

# --------------------------------------- the 1500 MHz block, re-measured -----
_rc = [r for r in load(os.path.join(D2, "E_recheck15.jsonl"))
       if "lat_median_ms" in r and "P_mean_W" in r]
if _rc:
    _rb = defaultdict(list)
    for r in _rc:
        _rb[(r["model"], r["threads"])].append(r)
    _models = sorted({m for m, _ in _rb})
    put("npRecheckNModels", str(len(_models)))
    _pw, _latdiff = [], []
    for m in _models:
        for thr in (2, 3, 4):
            v = _rb.get((m, thr))
            if v:
                _pw.append(med([r["P_mean_W"] for r in v]))
        # how well the re-run reproduces the original service times
        for thr in (1, 2, 3, 4):
            v = _rb.get((m, thr))
            o = by.get(1500000, {}).get(m, {}).get(thr, [])
            if v and o:
                a = med([r["lat_median_ms"] for r in v])
                b = med([r["lat_median_ms"] for r in o])
                _latdiff.append(100 * abs(a - b) / b)
    if _pw:
        put("npRecheckPowLo", f"{min(_pw):.2f}")
        put("npRecheckPowHi", f"{max(_pw):.2f}")
        put("npRecheckPowSpreadPct", f"{100*(max(_pw)-min(_pw))/min(_pw):.0f}")
    if _latdiff:
        put("npRecheckLatMaxPct", f"{max(_latdiff):.1f}")
    _one = [med([r["P_mean_W"] for r in _rb[(m, 1)]])
            for m in _models if _rb.get((m, 1))]
    if _one and _pw:
        put("npRecheckPowOne", f"{st.median(_one):.2f}")

# ---------------------------------------------------- the DDR rail ----------
# The PMIC reports a DDR rail. It is too small to be the DRAM array supply, so
# the manuscript reports it as a bound rather than as a mechanism.
if th:
    _d = defaultdict(list)
    for r in th:
        if r["freq_khz"] == K and r.get("P_ddr_mean_W") is not None:
            _d[r["threads"]].append(r)
    if _d.get(1) and _d.get(4):
        d1 = med([r["P_ddr_mean_W"] for r in _d[1]])
        d4 = med([r["P_ddr_mean_W"] for r in _d[4]])
        t1 = med([r["P_mean_W"] for r in _d[1]])
        t4 = med([r["P_mean_W"] for r in _d[4]])
        put("npDdrPowOne", f"{d1:.3f}")
        put("npDdrPowFour", f"{d4:.3f}")
        put("npDdrShareFour", f"{100*d4/t4:.1f}")
        put("npDdrGrowthPct", f"{100*(d4/d1-1):.0f}")
        put("npDdrOfDynamicPct", f"{100*(d4-d1)/(t4-t1):.1f}")
_di = [r.get("P_ddr_mean_W") for r in idle if r.get("P_ddr_mean_W") is not None]
if _di:
    put("npDdrFloorMw", f"{1000*med(_di):.1f}")



# ------------------------------- what the added instructions actually are ----
# perf2 showed instructions per inference growing with thread count on a fixed
# graph. This stage repeats the counters with the runtime's spin-waiting thread
# pool disabled, which turns the attribution from an inference into a
# measurement.
_ps = [r for r in load(os.path.join(D2, "E_perfspin.jsonl")) if r.get("counters")]
if _ps:
    _pb2 = defaultdict(list)
    for r in _ps:
        _pb2[(r["model"], r["threads"], r["spin"])].append(r)

    def _pm(model, thr, spin, field):
        v = _pb2.get((model, thr, spin), [])
        n = med([r["run"]["n_iter"] for r in v])
        c = med([r["counters"].get(field) for r in v])
        return (c / n) if (c and n) else None

    def _plat(model, thr, spin):
        v = _pb2.get((model, thr, spin), [])
        return med([r["run"]["ms_per_inf"] for r in v])

    for model, tag in (("resnet18-v1-7", "Big"), ("mnv3-cifar", "Mnv")):
        i1 = _pm(model, 1, "1", "instructions")
        i4 = _pm(model, 4, "1", "instructions")
        i4n = _pm(model, 4, "0", "instructions")
        if i1 and i4 and i4n:
            put("npSpinShare" + tag, f"{100*(i4-i4n)/(i4-i1):.0f}")
            put("npInstrNoSpin" + tag, f"{100*(i4n/i1-1):.0f}")
        l1_, l4_ = (_pm(model, 1, "1", "l2d_cache_refill"),
                    _pm(model, 4, "1", "l2d_cache_refill"))
        l1n, l4n = (_pm(model, 1, "0", "l2d_cache_refill"),
                    _pm(model, 4, "0", "l2d_cache_refill"))
        if l1_ and l4_ and l1n and l4n:
            put("npLTwoSpinOn" + tag, f"{100*(l4_/l1_-1):.0f}")
            put("npLTwoSpinOff" + tag, f"{100*(l4n/l1n-1):.0f}")
        a, b = _plat(model, 4, "1"), _plat(model, 4, "0")
        if a and b:
            put("npSpinLatCost" + tag, f"{100*(b/a-1):.0f}")
        c1, c4 = (_pm(model, 4, "1", "cycles"), _pm(model, 4, "0", "cycles"))
        if c1 and c4 and i4 and i4n:
            put("npIpcSpinOn" + tag, f"{i4/c1:.2f}")
            put("npIpcSpinOff" + tag, f"{i4n/c4:.2f}")



# ------------------------------- the floor at which each network breaks even -
# The two-floor sensitivity check answers "does the verdict hold at these two
# floors". The stronger question is what floor it would take to change the
# verdict, which the measurement answers directly: setting E_n = E_1 and
# solving for the floor gives P0* = (P1 t1 - Pn tn) / (t1 - tn). Any real floor
# above P0* means threading costs dynamic energy.
if by.get(K):
    stars = {}
    for m in CNN8 + ["mnv3-cifar"]:
        t1 = cell(K, m, 1, "lat_median_ms")
        t4 = cell(K, m, 4, "lat_median_ms")
        p1 = cell(K, m, 1, "P_mean_W")
        p4 = cell(K, m, 4, "P_mean_W")
        if None in (t1, t4, p1, p4) or t1 == t4:
            continue
        stars[m] = (p1 * t1 - p4 * t4) / (t1 - t4)
    if stars:
        cnn = {m: v for m, v in stars.items() if m in CNN8}
        put("npBreakFloorLo", f"{min(cnn.values()):.2f}")
        put("npBreakFloorHi", f"{max(cnn.values()):.2f}")
        put("npBreakFloorMed", f"{st.median(list(cnn.values())):.2f}")
        put("npBreakFloorMax", SHORT.get(max(cnn, key=cnn.get)))
        if K in P0:
            put("npBreakFloorMarginPct",
                f"{100*(P0[K]/max(cnn.values())-1):.0f}")
            put("npBreakFloorNBelow",
                str(sum(1 for v in cnn.values() if v < P0[K])))
        put("npBreakFloorMnv", f"{stars['mnv3-cifar']:.2f}"
            if "mnv3-cifar" in stars else None)

# ------------------- between-campaign reproducibility of the DYNAMIC ratio ---
# The manuscript bounded its comparisons with the total-energy discrepancy
# between two campaigns. Subtracting the floor amplifies that, so the dynamic
# figure is the one the dynamic verdicts have to be read against.
_gov2 = [r for r in load(os.path.join(D2, "E_governor.jsonl"))
         if r.get("gov") == "performance" and "lat_median_ms" in r]
if _gov2 and by.get(K) and K in P0:
    _gg = defaultdict(lambda: defaultdict(list))
    for r in _gov2:
        _gg[r["model"]][r["threads"]].append(r)
    _dd, _tt = [], []
    for m, d in _gg.items():
        if not (1 in d and 4 in d):
            continue
        t1 = med([r["lat_median_ms"] for r in d[1]])
        t4 = med([r["lat_median_ms"] for r in d[4]])
        p1 = med([r["P_mean_W"] for r in d[1]])
        p4 = med([r["P_mean_W"] for r in d[4]])
        ref = ratios(K, m, 4)
        if not ref or not ref.get("Edyn") or p1 <= P0[K]:
            continue
        e_dyn = ((p4 - P0[K]) / (p1 - P0[K])) / (t1 / t4)
        _dd.append(abs(e_dyn - ref["Edyn"]) / ref["Edyn"])
        if ref.get("Etot"):
            _tt.append(abs((p4 / p1) / (t1 / t4) - ref["Etot"]) / ref["Etot"])
    if _dd:
        put("npCampaignDynPct", f"{100*max(_dd):.0f}")
    if _tt:
        put("npCampaignTotPct", f"{100*max(_tt):.0f}")

# ------------------------- where the floor's dispersion actually lives -------
# The dynamic threshold subtracts the floor, so the floor's dispersion sets the
# threshold's. On this board that dispersion is entirely off the core rail.
if idle:
    _tot, _core = defaultdict(list), defaultdict(list)
    for r in idle:
        _tot[r["freq_khz"]].append(r["P_mean_W"])
        if r.get("P_core_mean_W") is not None:
            _core[r["freq_khz"]].append(r["P_core_mean_W"])
    _tsp = [100 * (max(v) - min(v)) / min(v) for v in _tot.values() if len(v) > 1]
    _csp = [100 * (max(v) - min(v)) / min(v) for v in _core.values() if len(v) > 1]
    if _tsp and _csp:
        put("npFloorSpreadTotal", f"{max(_tsp):.0f}")
        put("npFloorSpreadCore", f"{max(_csp):.0f}")

# ---------------------------------------- which models are re-exports --------
REEXPORT = ["resnet18-v1-7", "resnet50-v1-7", "densenet-12", "mobilenetv2-12"]
put("npReexportList",
    ", ".join(SHORT.get(m, m) for m in REEXPORT[:-1]) + " and "
    + SHORT.get(REEXPORT[-1], REEXPORT[-1]))
if by.get(K):
    _re = [ratios(K, m, 4)["S"] for m in REEXPORT if ratios(K, m, 4)]
    _st = [ratios(K, m, 4)["S"] for m in CNN8
           if m not in REEXPORT and ratios(K, m, 4)]
    if _re and _st:
        put("npReexportSpeed", f"{st.mean(_re):.2f}")
        put("npStockSpeed", f"{st.mean(_st):.2f}")



# ------------------------------ the slow mode, stated the way it behaves -----
# Block structure: within the main sweep one repetition index is slow for the
# whole nine-model block at a clock, which is a machine state rather than an
# independent per-cell event.
_thr_rows = [r for r in load(os.path.join(D2, "E_threads.jsonl"))
             if "lat_median_ms" in r]
if _thr_rows:
    _cells = defaultdict(list)
    for r in _thr_rows:
        _cells[(r["freq_khz"], r["model"], r["threads"])].append(r)
    _slow_by_clock = defaultdict(lambda: defaultdict(int))
    _coldest, _affected = 0, 0
    for k, v in _cells.items():
        if len(v) < 3:
            continue
        m = st.median([x["lat_median_ms"] for x in v])
        hit = [x for x in v if x["lat_median_ms"] > 1.25 * m]
        if not hit:
            continue
        _affected += 1
        for x in hit:
            _slow_by_clock[k[0]][x["rep"]] += 1
        temps = [x.get("temp_before_c") for x in v]
        if all(x is not None for x in temps):
            if min(temps) == min(x.get("temp_before_c") for x in hit):
                _coldest += 1
    for khz, tag in ((1500000, "Fifteen"), (2400000, "TwentyFour")):
        d = _slow_by_clock.get(khz)
        if d:
            put("npBimodalRep" + tag, str(max(d, key=d.get)))
    if _affected:
        put("npBimodalColdestFrac", f"{_coldest} of {_affected}")

# Denominators restricted to the cells that could show it, since the mode never
# appears at one or two threads and counting those flatters the control.
_elig_cells = _elig_hits = 0
for tag, fn, kf in (("Rerun", "E_threads3.jsonl",
                     lambda r: (r["freq_khz"], r["model"], r["threads"])),
                    ("Fifteen", "E_recheck15.jsonl",
                     lambda r: (r["model"], r["threads"])),
                    ("Pinned", "E_isolate.jsonl",
                     lambda r: (r["model"], r["threads"], r["pinned"]))):
    rows = [r for r in load(os.path.join(D2, fn))
            if "lat_median_ms" in r and r.get("threads") in (3, 4)]
    b = defaultdict(list)
    for r in rows:
        b[kf(r)].append(r)
    for k, v in b.items():
        if len(v) < 3:
            continue
        _elig_cells += 1
        m = st.median([x["lat_median_ms"] for x in v])
        if any(x["lat_median_ms"] > 1.25 * m for x in v):
            _elig_hits += 1
put("npBimodalQuietEligible", str(_elig_cells))
put("npBimodalQuietEligibleHits", str(_elig_hits))

# ---------------------------------- how close the GEMM clears actually are ---
_gm2 = [r for r in load(os.path.join(D2, "E_gemm.jsonl")) if "ms_median" in r]
if _gm2 and K in P0:
    _gb2 = defaultdict(lambda: defaultdict(list))
    for r in _gm2:
        _gb2[r.get("n") or r.get("size")][r["threads"]].append(r)
    _marg = []
    for n, d in _gb2.items():
        if 1 not in d:
            continue
        t1 = med([r["ms_median"] for r in d[1]])
        p1 = med([r["P_mean_W"] for r in d[1]])
        for thr, rs in d.items():
            if thr == 1 or p1 <= P0[K]:
                continue
            tn = med([r["ms_median"] for r in rs])
            pn = med([r["P_mean_W"] for r in rs])
            e = ((pn - P0[K]) / (p1 - P0[K])) / (t1 / tn)
            if e < 1:
                _marg.append(100 * (1 - e))
    if _marg:
        _marg.sort()
        put("npGemmClearMargins",
            ", ".join(f"{x:.1f}" for x in _marg[:-1]) + f" and {_marg[-1]:.1f}")
        put("npGemmClearMarginMax", f"{max(_marg):.1f}")



# ============================ round 6: the objections that needed measuring ===

# ---- the window model, tested against the deployment it describes -----------
# Every power term in this paper is measured back to back, which is the regime
# the window model distinguishes itself from. This runs the duty-cycled
# deployment directly and compares the measured whole-window energy ratio
# against what Equation (3) predicts from the saturated numbers.
_dt = [r for r in load(os.path.join(D2, "E_duty.jsonl"))
       if r.get("model") != "__idle__" and "P_mean_W" in r]
_di2 = [r for r in load(os.path.join(D2, "E_duty.jsonl"))
        if r.get("model") == "__idle__" and "P_mean_W" in r]
if _dt and _di2 and by.get(K):
    P0duty = med([r["P_mean_W"] for r in _di2])
    put("npDutyFloor", f"{P0duty:.2f}")
    db = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in _dt:
        db[r["model"]][r["period_ms"]][r["threads"]].append(r)
    errs, meas_all, rows_d = [], [], []
    for m in sorted(db):
        for per in sorted(db[m]):
            d = db[m][per]
            if not (1 in d and 4 in d):
                continue
            e1 = med([r["energy_per_inf_mJ"] for r in d[1]])
            e4 = med([r["energy_per_inf_mJ"] for r in d[4]])
            t1 = cell(K, m, 1, "lat_median_ms")
            t4 = cell(K, m, 4, "lat_median_ms")
            p1 = cell(K, m, 1, "P_mean_W")
            p4 = cell(K, m, 4, "P_mean_W")
            if None in (t1, t4, p1, p4):
                continue
            T = per / 1000.0
            pred = ((P0duty * T + (p4 - P0duty) * t4 / 1000.0) /
                    (P0duty * T + (p1 - P0duty) * t1 / 1000.0))
            meas = e4 / e1
            errs.append(100 * (meas / pred - 1))
            meas_all.append(meas)
            rows_d.append((m, per, meas, pred))
    if errs:
        put("npDutyNCells", str(len(errs)))
        put("npDutyErrMax", f"{max(abs(e) for e in errs):.1f}")
        put("npDutyErrMed", f"{st.median([abs(e) for e in errs]):.1f}")
        put("npDutyNSave", str(sum(1 for x in meas_all if x < 1)))
        put("npDutyRatioLo", f"{min(meas_all):.3f}")
        put("npDutyRatioHi", f"{max(meas_all):.3f}")
        put("npDutyPeriods", " and ".join(
            str(p) for p in sorted({p for _, p, _, _ in rows_d})))
        best = min(rows_d, key=lambda r: abs(r[2] / r[3] - 1))
        put("npDutyBestModel", SHORT.get(best[0], best[0]))
        put("npDutyBestPeriod", str(best[1]))
        put("npDutyBestMeas", f"{best[2]:.3f}")
        put("npDutyBestPred", f"{best[3]:.3f}")
        _dc = [r.get("duty_cycle") for r in _dt if r.get("duty_cycle")]
        if _dc:
            put("npDutyCycleLo", f"{100*min(_dc):.1f}")
            put("npDutyCycleHi", f"{100*max(_dc):.0f}")

# ---- does the verdict survive with the runtime's spin-wait switched off? ----
_se = [r for r in load(os.path.join(D2, "E_spinE.jsonl")) if "lat_median_ms" in r]
_sei = [r for r in load(os.path.join(D2, "E_idleE.jsonl")) if "P_mean_W" in r]
if _se and _sei:
    P0e = med([r["P_mean_W"] for r in _sei])
    put("npSpinEFloor", f"{P0e:.2f}")
    sb2 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in _se:
        sb2[r["model"]][r["spin"]][r["threads"]].append(r)

    def _e(m, sp, thr):
        d = sb2[m][sp]
        if 1 not in d or thr not in d:
            return None
        g = lambda t, f: med([x[f] for x in d[t]])
        p1 = g(1, "P_mean_W")
        if p1 <= P0e:
            return None
        S = g(1, "lat_median_ms") / g(thr, "lat_median_ms")
        return {"S": S, "Edyn": ((g(thr, "P_mean_W") - P0e) / (p1 - P0e)) / S,
                "t": g(thr, "lat_median_ms")}
    for thr, tag in ((2, "Two"), (4, "Four")):
        for sp, stag in (("1", "On"), ("0", "Off")):
            n = sum(1 for m in CNN8
                    if _e(m, sp, thr) and _e(m, sp, thr)["Edyn"] < 1)
            put("npSpinEN" + tag + stag, str(n))
    # the configurations that reach break-even once the pool stops spinning
    clears = []
    for m in CNN8:
        r2 = _e(m, "0", 2)
        if r2 and r2["Edyn"] < 1:
            clears.append((SHORT.get(m, m), 100 * (1 - r2["Edyn"])))
    if clears:
        clears.sort(key=lambda x: -x[1])
        put("npSpinEClearList",
            ", ".join(c[0] for c in clears[:-1]) + " and " + clears[-1][0])
        put("npSpinEClearMargins",
            ", ".join(f"{c[1]:.1f}" for c in clears[:-1])
            + f" and {clears[-1][1]:.1f}")
        put("npSpinEClearMarginMax", f"{max(c[1] for c in clears):.1f}")
    # what turning the pool off costs in latency at two threads
    lat = []
    for m in CNN8:
        a, b_ = _e(m, "1", 2), _e(m, "0", 2)
        if a and b_:
            lat.append(100 * (b_["t"] / a["t"] - 1))
    if lat:
        put("npSpinELatLo", f"{min(lat):.1f}")
        put("npSpinELatHi", f"{max(lat):.1f}")
    # medians at four threads, to show the penalty shrinks but does not vanish
    for sp, stag in (("1", "On"), ("0", "Off")):
        v = [_e(m, sp, 4)["Edyn"] for m in CNN8 if _e(m, sp, 4)]
        if v:
            put("npSpinEMedFour" + stag, f"{st.median(v):.2f}")

# ---- what the firmware delivers, as opposed to what cpufreq reports --------
_cv = load(os.path.join(D2, "E_clockv.jsonl"))
if _cv:
    hz = defaultdict(list)
    volts = defaultdict(list)
    thr_flags = set()
    for r in _cv:
        if r.get("arm_hz_med"):
            hz[r["freq_khz"]].append(r["arm_hz_med"])
        for f in ("core_v_min", "core_v_max"):
            if r.get(f) is not None:
                volts[r["freq_khz"]].append(r[f])
        for x in r.get("throttled", []):
            thr_flags.add(x)
    for khz, tag in ((1500000, "Fifteen"), (2400000, "TwentyFour")):
        if hz.get(khz):
            put("npClockArm" + tag, f"{st.median(hz[khz])/1e6:.0f}")
        if volts.get(khz):
            put("npClockVolt" + tag, f"{st.median(volts[khz]):.3f}")
            put("npClockVoltSpread" + tag,
                f"{max(volts[khz]) - min(volts[khz]):.3f}")
    put("npClockThrottled",
        "no throttling was reported at any point"
        if thr_flags == {"throttled=0x0"} else "throttling was reported")
    put("npClockNCells", str(len(_cv)))

# ---- the sweep again, in randomised order ----------------------------------
_sh = [r for r in load(os.path.join(D2, "E_shuffle.jsonl"))
       if "lat_median_ms" in r and "P_mean_W" in r]
if _sh and K in P0:
    hb = defaultdict(lambda: defaultdict(list))
    for r in _sh:
        hb[r["model"]][r["threads"]].append(r)
    n_save, deltas = 0, []
    for m in CNN8:
        d = hb[m]
        if 1 not in d or 4 not in d:
            continue
        g = lambda t, f: med([x[f] for x in d[t]])
        p1 = g(1, "P_mean_W")
        if p1 <= P0[K]:
            continue
        S = g(1, "lat_median_ms") / g(4, "lat_median_ms")
        e = ((g(4, "P_mean_W") - P0[K]) / (p1 - P0[K])) / S
        n_save += e < 1
        ref = ratios(K, m, 4)
        if ref and ref.get("Edyn"):
            deltas.append(100 * abs(e / ref["Edyn"] - 1))
    put("npShuffleNSave", str(n_save))
    if deltas:
        put("npShuffleDeltaMax", f"{max(deltas):.0f}")
    cells = hits = 0
    for m in hb:
        for th in hb[m]:
            v = [x["lat_median_ms"] for x in hb[m][th]]
            if len(v) < 3:
                continue
            cells += 1
            if max(v) > 1.25 * st.median(v):
                hits += 1
    put("npShuffleCells", str(cells))
    put("npShuffleSlow", str(hits))

# ---- counters from a slope rather than a two-point difference --------------
_pn = load(os.path.join(D2, "E_perfN.jsonl"))
if _pn:
    nb = defaultdict(lambda: defaultdict(list))
    for r in _pn:
        nb[(r["model"], r.get("res"), r["threads"])][r["n_iter"]].append(r)
    slopes, r2s, agree = {}, [], []
    for k, d in nb.items():
        ns = sorted(d)
        ys = [med([r["counters"]["instructions"] for r in d[n]]) for n in ns]
        n = len(ns)
        sx, sy = sum(ns), sum(ys)
        sxx = sum(x * x for x in ns)
        sxy = sum(x * y for x, y in zip(ns, ys))
        if n * sxx - sx * sx == 0:
            continue
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        icept = (sy - slope * sx) / n
        ybar = sy / n
        sst = sum((y - ybar) ** 2 for y in ys)
        ssr = sum((y - (slope * x + icept)) ** 2 for x, y in zip(ns, ys))
        if sst:
            r2s.append(1 - ssr / sst)
        slopes[k] = slope
        big = max(ns)
        two = (ys[ns.index(big)] - ys[ns.index(0)]) / big if 0 in ns else None
        if two:
            agree.append(100 * abs(two / slope - 1))
    if r2s:
        put("npPerfNRsq", f"{min(r2s):.5f}")
    if agree:
        put("npPerfNAgree", f"{max(agree):.1f}")
    for key, tag in ((("resnet18-v1-7", 224), "Big"),
                     (("resnet18-v1-7", 32), "Small"),
                     (("mnv3-cifar", None), "Mnv")):
        a = slopes.get((key[0], key[1], 1))
        b_ = slopes.get((key[0], key[1], 4))
        if a and b_:
            put("npSlopeGrowth" + tag, f"{100*(b_/a-1):.0f}")



# ================= round 7/8: intervals, and honest denominators =============

# ---- bootstrap intervals on the quantity the verdict rests on ---------------
# Ten repetitions of every one- and four-thread cell at 2400 MHz, in randomised
# order, with the floor measured in the same session. Enough to replace "we
# quote no confidence intervals" with an actual interval.
import random as _rnd

_ci = [r for r in load(os.path.join(D2, "E_ci.jsonl"))
       if "lat_median_ms" in r and "P_mean_W" in r]
_ici = [r for r in load(os.path.join(D2, "E_idleci.jsonl"))
        if r.get("freq_khz") == K and "P_mean_W" in r]
if _ci and _ici:
    _cb = defaultdict(lambda: defaultdict(list))
    for r in _ci:
        _cb[r["model"]][r["threads"]].append(r)
    _fl = [r["P_mean_W"] for r in _ici]
    put("npCiFloor", f"{med(_fl):.2f}")
    put("npCiFloorLo", f"{min(_fl):.2f}")
    put("npCiFloorHi", f"{max(_fl):.2f}")
    put("npCiReps", str(min(len(v[1]) for v in _cb.values() if 1 in v)))

    def _boot(model, n_boot=4000, seed=11):
        d = _cb[model]
        if 1 not in d or 4 not in d:
            return None
        t1 = [x["lat_median_ms"] for x in d[1]]
        t4 = [x["lat_median_ms"] for x in d[4]]
        p1 = [x["P_mean_W"] for x in d[1]]
        p4 = [x["P_mean_W"] for x in d[4]]
        rng = _rnd.Random(seed)
        out = []
        for _ in range(n_boot):
            a = med([rng.choice(t1) for _ in t1])
            b = med([rng.choice(t4) for _ in t4])
            c = med([rng.choice(p1) for _ in p1])
            e = med([rng.choice(p4) for _ in p4])
            f = med([rng.choice(_fl) for _ in _fl])
            if c <= f:
                continue
            out.append(((e - f) / (c - f)) / (a / b))
        out.sort()
        if len(out) < 100:
            return None
        return (out[int(0.025 * len(out))], med(out), out[int(0.975 * len(out))])

    _res = {}
    for m in CNN8:
        r = _boot(m)
        if r:
            _res[m] = r
    if _res:
        put("npCiNModels", str(len(_res)))
        # how many have their whole interval above break-even
        _above = [m for m, v in _res.items() if v[0] > 1.0]
        put("npCiNAbove", str(len(_above)))
        _lo = min(v[0] for v in _res.values())
        _hi = max(v[2] for v in _res.values())
        put("npCiRange", f"{_lo:.2f}--{_hi:.2f}")
        _cm = min(_res, key=lambda m: _res[m][0])
        put("npCiClosest", SHORT.get(_cm, _cm))
        put("npCiClosestLo", f"{_res[_cm][0]:.2f}")
        put("npCiClosestMed", f"{_res[_cm][1]:.2f}")
        put("npCiClosestHi", f"{_res[_cm][2]:.2f}")
        _w = [100 * (v[2] - v[0]) / v[1] for v in _res.values()]
        put("npCiWidthMed", f"{st.median(_w):.0f}")

# ---- the floor is not the same number in every campaign --------------------
_floors = {}
for tag, fn, filt in (("Main", "E_idle.jsonl",
                       lambda r: r.get("freq_khz") == K
                       and r.get("mode") in ("idle", "idle_after_load")),
                      ("SpinE", "E_idleE.jsonl", lambda r: True),
                      ("Duty", "E_duty.jsonl",
                       lambda r: r.get("model") == "__idle__"),
                      ("Ci", "E_idleci.jsonl",
                       lambda r: r.get("freq_khz") == K)):
    v = [r["P_mean_W"] for r in load(os.path.join(D2, fn))
         if "P_mean_W" in r and filt(r)]
    if v:
        _floors[tag] = med(v)
if len(_floors) > 2:
    put("npFloorAcrossLo", f"{min(_floors.values()):.2f}")
    put("npFloorAcrossHi", f"{max(_floors.values()):.2f}")
    put("npFloorAcrossN", str(len(_floors)))
    put("npFloorAcrossPct",
        f"{100*(max(_floors.values())/min(_floors.values())-1):.0f}")

# ---- between-campaign spread, over EVERY replication we have ---------------
# An earlier version of this analysis quoted the governor campaign alone, which
# is the smallest of the four and understated the figure by half.
def _dyn_of(rows, floor, model, thr=4):
    d = defaultdict(list)
    for r in rows:
        if r.get("model") == model and "lat_median_ms" in r:
            d[r["threads"]].append(r)
    if 1 not in d or thr not in d:
        return None
    g = lambda t, f: med([x[f] for x in d[t]])
    p1 = g(1, "P_mean_W")
    if not p1 or p1 <= floor:
        return None
    S = g(1, "lat_median_ms") / g(thr, "lat_median_ms")
    return ((g(thr, "P_mean_W") - floor) / (p1 - floor)) / S


_reps_camp = [
    ("governor", [r for r in load(os.path.join(D2, "E_governor.jsonl"))
                  if r.get("gov") == "performance"], P0.get(K)),
    ("shuffle", load(os.path.join(D2, "E_shuffle.jsonl")), P0.get(K)),
    ("spin-wait", [r for r in load(os.path.join(D2, "E_spinE.jsonl"))
                   if r.get("spin") == "1"],
     med([r["P_mean_W"] for r in load(os.path.join(D2, "E_idleE.jsonl"))
          if "P_mean_W" in r])),
    ("repetition", load(os.path.join(D2, "E_ci.jsonl")),
     med([r["P_mean_W"] for r in load(os.path.join(D2, "E_idleci.jsonl"))
          if r.get("freq_khz") == K and "P_mean_W" in r])),
]
_devs, _worst = [], None
for tag, rows, fl in _reps_camp:
    if not rows or not fl:
        continue
    for m in CNN8:
        a = _dyn_of(rows, fl, m)
        ref = ratios(K, m, 4)
        if a and ref and ref.get("Edyn"):
            d = 100 * abs(a / ref["Edyn"] - 1)
            _devs.append(d)
            if _worst is None or d > _worst[0]:
                _worst = (d, SHORT.get(m, m), tag)
if _devs:
    put("npCampaignDynPct", f"{max(_devs):.0f}")
    put("npCampaignDynMed", f"{st.median(_devs):.0f}")
    put("npCampaignNReps", str(len([1 for _, r, f in _reps_camp if r and f])))
    if _worst:
        put("npCampaignWorstModel", _worst[1])
        put("npCampaignWorstCamp", _worst[2])

# ---- the sign flip the earlier analysis reported only as a count -----------
_se2 = [r for r in load(os.path.join(D2, "E_spinE.jsonl"))
        if r.get("spin") == "1"]
_fl2 = med([r["P_mean_W"] for r in load(os.path.join(D2, "E_idleE.jsonl"))
            if "P_mean_W" in r])
if _se2 and _fl2:
    _flips = []
    for m in CNN8:
        a = _dyn_of(_se2, _fl2, m, thr=2)
        ref = ratios(K, m, 2)
        if a and ref and ref.get("Edyn") and (a < 1) != (ref["Edyn"] < 1):
            _flips.append((SHORT.get(m, m), ref["Edyn"], a))
    if _flips:
        _f = _flips[0]
        put("npFlipModel", _f[0])
        put("npFlipSweep", f"{_f[1]:.2f}")
        put("npFlipOther", f"{_f[2]:.2f}")
        put("npFlipSavePct", f"{100*(1-_f[2]):.0f}")
        put("npFlipN", str(len(_flips)))


# the spin-wait energy campaign shows the slow mode too, and it was launched
# the same way the quiet campaigns were, which is what refutes the queueing
# explanation an earlier version of this analysis offered
_sec = _seh = 0
_seb = defaultdict(list)
for r in load(os.path.join(D2, "E_spinE.jsonl")):
    if "lat_median_ms" in r and r.get("threads") in (3, 4):
        _seb[(r["model"], r["threads"], r["spin"])].append(r["lat_median_ms"])
for _k, _v in _seb.items():
    if len(_v) < 3:
        continue
    _sec += 1
    if max(_v) > 1.25 * st.median(_v):
        _seh += 1
put("npBimodalSpinECells", f"{_seh} of {_sec}")


# ---- does the instrument perturb what it measures? -------------------------
# The sampler runs on the machine under test. Rather than assert a direction
# for its bias, vary how often it runs and see whether the answer moves.
_sr = [r for r in load(os.path.join(D2, "E_samplerate.jsonl"))
       if "P_mean_W" in r]
if _sr:
    _sb2 = defaultdict(list)
    for r in _sr:
        _sb2[(r["threads"], r["period_ms"])].append(r)
    _rates, _spread = {}, {}
    for th in (1, 4):
        ps = [(p, med([x["P_mean_W"] for x in v]))
              for (t_, p), v in _sb2.items() if t_ == th]
        ls = [med([x["lat_median_ms"] for x in v])
              for (t_, p), v in _sb2.items() if t_ == th]
        rr = [med([x["n_pmic_samples"] / x["pmic_span_s"] for x in v])
              for (t_, p), v in _sb2.items() if t_ == th]
        if len(ps) > 1:
            vals = [v for _, v in ps]
            _spread[th] = 100 * (max(vals) - min(vals)) / min(vals)
            _rates[th] = (min(rr), max(rr))
            put("npSampleSpread" + ("One" if th == 1 else "Four"),
                f"{_spread[th]:.1f}")
            put("npSampleLatSpread" + ("One" if th == 1 else "Four"),
                f"{100*(max(ls)-min(ls))/min(ls):.1f}")
    if _rates:
        lo = min(r[0] for r in _rates.values())
        hi = max(r[1] for r in _rates.values())
        put("npSampleRateLo", f"{lo:.1f}")
        put("npSampleRateHi", f"{hi:.0f}")
        put("npSampleRateFactor", f"{hi/lo:.0f}")

# ---- what a live thread pool costs while the device does nothing -----------
_si = [r for r in load(os.path.join(D2, "E_spinidle.jsonl"))
       if "P_mean_W" in r]
if _si:
    _sib = defaultdict(dict)
    for r in _si:
        k = "none" if r["threads"] == 0 else "t%d/s%s" % (r["threads"],
                                                          r["spin"])
        _sib[r["rep"]][k] = r["P_mean_W"]
    for th in (1, 4):
        d = [_sib[r]["t%d/s1" % th] - _sib[r]["t%d/s0" % th]
             for r in _sib
             if "t%d/s1" % th in _sib[r] and "t%d/s0" % th in _sib[r]]
        if d:
            tag = "One" if th == 1 else "Four"
            put("npSpinIdleDelta" + tag, f"{1000*st.mean(d):.0f}")
            put("npSpinIdlePos" + tag,
                "%d of %d" % (sum(1 for x in d if x > 0), len(d)))
    _nones = [_sib[r]["none"] for r in _sib if "none" in _sib[r]]
    if len(_nones) > 1:
        put("npSpinIdleDriftPct",
            f"{100*(max(_nones)-min(_nones))/min(_nones):.0f}")

# ---- the floor, watched for long enough to see it move ---------------------
_fd = [r for r in load(os.path.join(D2, "E_floordrift.jsonl"))
       if "P_mean_W" in r]
if len(_fd) > 3:
    _fd.sort(key=lambda r: r.get("idx", 0))
    _pw = [r["P_mean_W"] for r in _fd]
    put("npDriftN", str(len(_fd)))
    put("npDriftMinutes", f"{max(r['elapsed_s'] for r in _fd)/60:.0f}")
    put("npDriftLo", f"{min(_pw):.2f}")
    put("npDriftHi", f"{max(_pw):.2f}")
    put("npDriftPct", f"{100*(max(_pw)-min(_pw))/min(_pw):.0f}")
    _t = [r.get("temp_end_c") for r in _fd if r.get("temp_end_c")]
    if _t:
        put("npDriftTempLo", f"{min(_t):.0f}")
        put("npDriftTempHi", f"{max(_t):.0f}")



# ===================== the duty-cycle test, done with matched work ==========
# Two earlier attempts got the comparison wrong: the first fixed the period but
# at duty cycles too low to discriminate, the second fixed the duty cycle,
# which makes the two arms serve different numbers of inferences. This fixes
# the period from the one-thread service time and uses it for both arms, so W
# and T are identical, at duty cycles where the model and a null predictor
# disagree.
_d3 = [r for r in load(os.path.join(D2, "E_duty3.jsonl"))
       if r.get("model") != "__idle__" and "P_mean_W" in r]
_d3i = [r for r in load(os.path.join(D2, "E_duty3.jsonl"))
        if r.get("model") == "__idle__" and "P_mean_W" in r]
if _d3 and _d3i and by.get(K):
    P0d3 = med([r["P_mean_W"] for r in _d3i])
    put("npDutyThreeFloor", f"{P0d3:.2f}")

    def _aliased(r):
        """The sampler locks to the workload period and can miss the burst.

        Unbiased integration needs the burst to be sampled. When the period is
        close to an integer multiple of the sampling interval the phase does
        not drift, and when the burst is shorter than one sampling interval the
        sampler can sit in the idle gap every period. Both conditions together
        make a cell unusable; either alone does not.
        """
        si = 1000.0 * r["pmic_span_s"] / r["n_pmic_samples"]
        ratio = r["period_ms"] / si
        return (abs(ratio - round(ratio)) < 0.06 and round(ratio) >= 1
                and r["lat_median_ms"] / si < 1.0)

    d3b = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    excluded = set()
    for r in _d3:
        d3b[r["model"]][r["one_thread_duty"]][r["threads"]].append(r)
        if _aliased(r):
            excluded.add((r["model"], r["one_thread_duty"]))
    errs, nulls, meas_all, duties = [], [], [], []
    for m in d3b:
        for d1 in d3b[m]:
            if (m, d1) in excluded:
                continue
            d = d3b[m][d1]
            if 1 not in d or 4 not in d:
                continue
            e1 = med([r["energy_per_inf_mJ"] for r in d[1]])
            e4 = med([r["energy_per_inf_mJ"] for r in d[4]])
            t1 = cell(K, m, 1, "lat_median_ms")
            t4 = cell(K, m, 4, "lat_median_ms")
            p1 = cell(K, m, 1, "P_mean_W")
            p4 = cell(K, m, 4, "P_mean_W")
            if None in (t1, t4, p1, p4):
                continue
            T = (t1 / 1000.0) / d1
            pred = ((P0d3 * T + (p4 - P0d3) * t4 / 1000.0) /
                    (P0d3 * T + (p1 - P0d3) * t1 / 1000.0))
            meas = e4 / e1
            errs.append(abs(100 * (meas / pred - 1)))
            nulls.append(abs(100 * (meas - 1)))
            meas_all.append(meas)
            duties.append(d1)
    if errs:
        put("npDutyThreeN", str(len(errs)))
        put("npDutyThreeExcluded", str(len(excluded)))
        put("npDutyThreeErrMed", f"{st.median(errs):.1f}")
        put("npDutyThreeErrMax", f"{max(errs):.1f}")
        put("npDutyThreeNullMed", f"{st.median(nulls):.1f}")
        put("npDutyThreeBeats", f"{st.median(nulls)/st.median(errs):.1f}")
        put("npDutyThreeRatioLo", f"{min(meas_all):.3f}")
        put("npDutyThreeRatioHi", f"{meas_all[meas_all.index(max(meas_all))]:.3f}")
        put("npDutyThreeNSave", str(sum(1 for x in meas_all if x < 1)))
        put("npDutyThreeDutyLo", f"{100*min(duties):.0f}")
        put("npDutyThreeDutyHi", f"{100*max(duties):.0f}")

# ================ the 1500 MHz anomaly, with the workload removed ===========
# N spinning processes, no runtime, no memory traffic, no allocator. If the
# anomaly reproduces here it is the board or the instrument, not the inference.
_pc = [r for r in load(os.path.join(D2, "E_purecore2.jsonl"))
       if "P_mean_W" in r]
if _pc:
    pb2 = defaultdict(list)
    for r in _pc:
        pb2[(r["freq_khz"], r["n_proc"])].append(r)
    for khz, tag in ((2400000, "TwentyFour"), (1500000, "Fifteen")):
        pw = {}
        it = {}
        for n in (0, 1, 2, 3, 4):
            v = pb2.get((khz, n))
            if v:
                pw[n] = med([x["P_mean_W"] for x in v])
                it[n] = med([x["iterations"] for x in v])
        if len(pw) == 5:
            deltas = [pw[n] - pw[n - 1] for n in (1, 2, 3, 4)]
            put("npPureLo" + tag, f"{min(deltas):+.2f}")
            put("npPureHi" + tag, f"{max(deltas):+.2f}")
            put("npPureNeg" + tag, str(sum(1 for d in deltas if d < 0)))
            put("npPureSpread" + tag,
                f"{100*(max(deltas)-min(deltas))/abs(st.mean(deltas)):.0f}")
            if it.get(4) and it.get(1):
                put("npPureScale" + tag, f"{it[4]/it[1]:.2f}")


put("npOrtVersion", "1.24.3")

# Any macro the manuscript uses but the campaign has not yet produced gets a
# visible placeholder, so the paper always compiles while data is landing.
import re as _re

_tex = os.path.join(ROOT, "paper-v2", "main.tex")
if os.path.exists(_tex):
    used = set(_re.findall(r"\\(np[A-Za-z]+)", io.open(_tex, encoding="utf-8").read()))
    for name in sorted(used):
        M.setdefault(name, r"\texttt{TBD}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("% generated by code/v2/fill_numbers_v2.py -- do not edit\n")
    for k in sorted(M):
        v = M[k]
        if k.startswith("npTable"):
            f.write("\\newcommand{\\%s}{%%\n%s}\n" % (k, v))
        else:
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print("wrote", OUT)
tbd = [k for k, v in M.items() if "TBD" in str(v)]
print(f"{len(M)} macros, {len(tbd)} still TBD" + (": " + ", ".join(sorted(tbd)) if tbd else ""))
