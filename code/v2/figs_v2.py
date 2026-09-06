#!/usr/bin/env python3
"""Figures for the revised paper.

The first submission carried no figures at all, which a reviewer objected to.
Four are generated here, each carrying an argument the tables cannot make
compactly.
"""

import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
# Type 3 fonts are matplotlib's default and an explicit desk-reject trigger at
# several IEEE venues; 42 selects TrueType.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D2 = os.path.join(ROOT, "data", "v2")
D1 = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 8, "font.family": "serif", "figure.dpi": 300})
ACC, WARN, INK, MUT = "#2b6cb0", "#c05621", "#1a1a1a", "#718096"

CNN8 = ["googlenet-12", "resnet18-v1-7", "efficientnet-lite4-11",
        "squeezenet1.1-7", "shufflenet-v2-10", "resnet50-v1-7",
        "mobilenetv2-12", "densenet-12"]
SHORT = {"googlenet-12": "GoogLeNet", "resnet18-v1-7": "ResNet-18",
         "efficientnet-lite4-11": "EffNet-Lite4", "squeezenet1.1-7": "SqueezeNet",
         "shufflenet-v2-10": "ShuffleNet-v2", "resnet50-v1-7": "ResNet-50",
         "mobilenetv2-12": "MobileNetV2", "densenet-12": "DenseNet-121",
         "mnv3-cifar": "MobileNetV3 (CIFAR)"}
K = 2400000


def load(p):
    out = []
    if os.path.exists(p):
        with open(p) as f:
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


idle = [r for r in load(os.path.join(D2, "E_idle.jsonl"))
        if r.get("mode") in ("idle", "idle_after_load") and "P_mean_W" in r]
P0 = defaultdict(list)
for r in idle:
    P0[r["freq_khz"]].append(r["P_mean_W"])
P0 = {k: med(v) for k, v in P0.items()}

# The three-thread cells were re-measured; the figures must read the same data
# the tables do, or Fig. 2 plots the contaminated repetitions while the table
# beside it plots the clean ones.
th = [r for r in load(os.path.join(D2, "E_threads3.jsonl"))
      if "lat_median_ms" in r and "P_mean_W" in r]
th += [r for r in load(os.path.join(D2, "E_threads.jsonl"))
      if "lat_median_ms" in r and "P_mean_W" in r]
T = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in th:
    T[r["freq_khz"]][r["model"]][r["threads"]].append(r)


def rat(khz, m, t):
    d = T[khz][m]
    if 1 not in d or t not in d:
        return None
    t1 = med([r["lat_median_ms"] for r in d[1]])
    tn = med([r["lat_median_ms"] for r in d[t]])
    p1 = med([r["P_mean_W"] for r in d[1]])
    pn = med([r["P_mean_W"] for r in d[t]])
    pi = P0.get(khz)
    s = t1 / tn
    out = {"S": s, "Ptot": pn / p1, "Etot": (pn / p1) / s}
    if pi and p1 > pi:
        out["Pdyn"] = (pn - pi) / (p1 - pi)
        out["Edyn"] = out["Pdyn"] / s
    return out


# ------------------------------------------------- Fig 1: break-even plane --
def fig_plane():
    if not T.get(K):
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    lim = 0
    for m in CNN8 + ["mnv3-cifar"]:
        r = rat(K, m, 4)
        if not r:
            continue
        ax.scatter(r["Ptot"], r["S"], color=ACC, s=26, zorder=3)
        if "Pdyn" in r:
            ax.scatter(r["Pdyn"], r["S"], color=WARN, s=26, marker="s", zorder=3)
            ax.plot([r["Ptot"], r["Pdyn"]], [r["S"], r["S"]], color=MUT,
                    lw=0.6, ls=":", zorder=2)
            lim = max(lim, r["Pdyn"], r["S"])
    # GEMM, if measured
    n_gemm = 0
    gm = [r for r in load(os.path.join(D2, "E_gemm.jsonl")) if "ms_median" in r]
    gb = defaultdict(lambda: defaultdict(list))
    for r in gm:
        gb[r["size"]][r["threads"]].append(r)
    for size in sorted(gb):
        d = gb[size]
        if 1 in d and 4 in d:
            t1 = med([r["ms_median"] for r in d[1]])
            t4 = med([r["ms_median"] for r in d[4]])
            p1 = med([r["P_mean_W"] for r in d[1]])
            p4 = med([r["P_mean_W"] for r in d[4]])
            pi = P0.get(K)
            if None in (t1, t4, p1, p4):
                continue
            ax.scatter(p4 / p1, t1 / t4, color="#2f855a", s=22, marker="^", zorder=3)
            n_gemm += 1
            if pi and p1 > pi:
                ax.scatter((p4 - pi) / (p1 - pi), t1 / t4, color="#2f855a", s=22,
                           marker="v", zorder=3)
                lim = max(lim, (p4 - pi) / (p1 - pi), t1 / t4)
    lim = max(lim, 3.9) + 0.3
    ax.plot([0, lim], [0, lim], color=INK, lw=1.0, ls="--")
    ax.text(lim * 0.70, lim * 0.74, "break-even", rotation=40, fontsize=6.5,
            color=INK)
    ax.text(1.08, lim * 0.94, "above the line: threads save energy",
            fontsize=6.2, color=MUT)
    ax.text(1.08, 0.62, "below the line: threads cost energy",
            fontsize=6.2, color=MUT)
    ax.set_xlim(1.0, lim)
    ax.set_ylim(0.5, lim)
    ax.set_xlabel("power ratio at four threads")
    ax.set_ylabel("parallel speedup $S_4$")
    ax.scatter([], [], color=ACC, s=26, label="CNN, total-energy")
    ax.scatter([], [], color=WARN, s=26, marker="s", label="CNN, dynamic")
    if n_gemm:
        ax.scatter([], [], color="#2f855a", s=22, marker="^",
                   label="GEMM, total-energy")
        ax.scatter([], [], color="#2f855a", s=22, marker="v",
                   label="GEMM, dynamic")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_plane.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "v2_plane.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote v2_plane.pdf")


# ----------------------------------------------------- Fig 2: thread scaling --
def fig_scaling():
    if not T.get(K):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for m in CNN8:
        xs, ys = [], []
        for t in (1, 2, 3, 4):
            r = rat(K, m, t)
            if r:
                xs.append(t)
                ys.append(r["S"])
        if xs:
            ax.plot(xs, ys, marker="o", ms=3, lw=1.0, label=SHORT.get(m, m))
    r = rat(K, "mnv3-cifar", 4)
    xs, ys = [], []
    for t in (1, 2, 3, 4):
        rr = rat(K, "mnv3-cifar", t)
        if rr:
            xs.append(t)
            ys.append(rr["S"])
    if xs:
        ax.plot(xs, ys, marker="s", ms=3.5, lw=1.4, color=INK, ls="--",
                label=SHORT["mnv3-cifar"])
    ax.plot([1, 4], [1, 4], color=MUT, lw=0.8, ls=":")
    ax.text(3.1, 3.5, "linear", fontsize=6.5, color=MUT, rotation=32)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup $S_n$")
    ax.set_title("(a) parallel speedup", fontsize=8)
    ax.legend(fontsize=5.6, ncol=2, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for m in CNN8:
        xs, yt, yd = [], [], []
        for t in (1, 2, 3, 4):
            r = rat(K, m, t)
            if r:
                xs.append(t)
                yt.append(r["Etot"])
                yd.append(r.get("Edyn", float("nan")))
        if xs:
            ax.plot(xs, yt, marker="o", ms=2.5, lw=0.9, color=ACC, alpha=0.75)
            ax.plot(xs, yd, marker="s", ms=2.5, lw=0.9, color=WARN, alpha=0.75)
    for t_ in (1, 2, 3, 4):
        r = rat(K, "mnv3-cifar", t_)
    xs, yt, yd = [], [], []
    for t_ in (1, 2, 3, 4):
        r = rat(K, "mnv3-cifar", t_)
        if r:
            xs.append(t_)
            yt.append(r["Etot"])
            yd.append(r.get("Edyn", float("nan")))
    if xs:
        ax.plot(xs, yt, marker="s", ms=3, lw=1.3, color=ACC, ls="--")
        ax.plot(xs, yd, marker="s", ms=3, lw=1.3, color=WARN, ls="--")
    ax.axhline(1.0, color=INK, lw=1.0, ls="--")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("threads")
    ax.set_ylabel("energy per inference, relative to 1 thread")
    ax.set_title("(b) energy under the two accountings", fontsize=8)
    ax.plot([], [], color=ACC, marker="o", ms=3, label="total energy")
    ax.plot([], [], color=WARN, marker="s", ms=3, label="dynamic energy")
    ax.plot([], [], color=MUT, ls="--", lw=1.3, label="CIFAR-scale (dashed)")
    ax.legend(fontsize=6.5, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_scaling.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "v2_scaling.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote v2_scaling.pdf")


# --------------------------------------------------------- Fig 3: counters --
def fig_perf():
    """Counters against thread count, with the spin-wait switched off.

    The first version of this figure read E_perf.jsonl, which counted the whole
    process including session construction; it is superseded by E_perf2.jsonl,
    which subtracts a startup-only run. The spin-off points come from
    E_perfspin.jsonl and are the figure's point: the instruction growth is the
    thread pool, the cache growth is not.
    """
    recs = [r for r in load(os.path.join(D2, "E_perf2.jsonl")) if r.get("counters")]
    if not recs:
        return
    pb = defaultdict(lambda: defaultdict(list))
    for r in recs:
        pb[(r["model"], r.get("res"))][r["threads"]].append(r)
    key = ("resnet18-v1-7", 224)
    if key not in pb:
        key = sorted(pb, key=str)[0]
    ths = sorted(pb[key])

    def per_inf(rs, name):
        c = med([r["counters"].get(name) for r in rs])
        n = med([r["run"].get("n_iter") for r in rs]) or 1
        return (c or 0) / n

    instr, l2, stall = [], [], []
    for th in ths:
        rs = pb[key][th]
        instr.append(per_inf(rs, "instructions") / 1e6)
        l2.append(per_inf(rs, "l2d_cache_refill") / 1e3)
        cyc = med([r["counters"].get("cycles") for r in rs]) or 1
        stl = med([r["counters"].get("stalled-cycles-backend") for r in rs]) or 0
        stall.append(100.0 * stl / cyc)

    # the same quantities with the runtime's spin-wait disabled, for both the
    # ImageNet-scale and the CIFAR-scale network
    spin_recs = load(os.path.join(D2, "E_perfspin.jsonl"))
    sp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in spin_recs:
        if r.get("counters"):
            sp[r["model"]][r["spin"]][r["threads"]].append(r)
    off_t = sorted(sp.get(key[0], {}).get("0", {}))
    off_l2 = [per_inf(sp[key[0]]["0"][th], "l2d_cache_refill") / 1e3
              for th in off_t]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    a = axes[0]
    # instructions per inference relative to one thread, so the vertical axis
    # starts at one and the size of the effect is unambiguous
    styles = {"resnet18-v1-7": ("o", "-"), "mnv3-cifar": ("^", "-.")}
    for model, (mk, ls) in styles.items():
        if model not in sp:
            continue
        base = per_inf(sp[model]["1"].get(1, []), "instructions")
        if not base:
            continue
        lbl = SHORT.get(model, model)
        for spin, colour, dash in (("1", ACC, ls), ("0", WARN, "--")):
            xs = sorted(sp[model][spin])
            ys = [per_inf(sp[model][spin][th], "instructions") / base
                  for th in xs]
            a.plot(xs, ys, marker=mk, ms=4, color=colour, lw=1.4, ls=dash,
                   label="%s, spin %s" % (lbl, "on" if spin == "1" else "off"))
    a.axhline(1.0, color=INK, lw=0.8, ls=":")
    a.set_xticks(ths)
    a.set_xlabel("threads")
    a.set_ylabel("instructions per inference, relative to 1 thread")
    a.set_title("(a) the added work is the thread pool", fontsize=8)
    a.legend(fontsize=5.8, framealpha=0.9)

    b = axes[1]
    b.plot(ths, l2, marker="o", ms=4, color=ACC, lw=1.4, label="spin-wait on")
    if off_t:
        b.plot(off_t, off_l2, marker="s", ms=4.5, color=WARN, lw=1.4, ls="--",
               label="spin-wait off")
    b.set_xticks(ths)
    b.set_xlabel("threads")
    b.set_ylabel("L2 refills per inference (thousands)", color=ACC)
    b2 = b.twinx()
    b2.plot(ths, stall, marker="^", ms=4, color=MUT, lw=1.2, ls=":",
            label="backend stalls")
    b2.set_ylabel("backend stall cycles (% of total)", color=MUT)
    b.set_title("(b) the cache contention is not", fontsize=8)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2l = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2l, fontsize=6.5, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_perf.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "v2_perf.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote v2_perf.pdf")


# ------------------------------------------------------- Fig 4: resolution --
def fig_resolution():
    recs = []
    for name in ("P15c_resolution.jsonl", "P15d_resolution_more.jsonl"):
        recs += load(os.path.join(D1, name))
    recs = [r for r in recs if "ms_per_inf" in r and "P_mean_W" in r
            and r.get("resolution")]
    if not recs:
        return
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        by[r["model"].replace(".onnx", "")][r["resolution"]][r["threads"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    pi = P0.get(K)
    for m in sorted(by):
        xs, s4, e4, ed = [], [], [], []
        for res in sorted(by[m]):
            d = by[m][res]
            if 1 not in d or 4 not in d:
                continue
            t1 = med([r["ms_per_inf"] for r in d[1]])
            t4 = med([r["ms_per_inf"] for r in d[4]])
            p1 = med([r["P_mean_W"] for r in d[1]])
            p4 = med([r["P_mean_W"] for r in d[4]])
            s = t1 / t4
            xs.append(res)
            s4.append(s)
            e4.append((p4 / p1) / s)
            ed.append(((p4 - pi) / (p1 - pi)) / s if pi and p1 > pi else np.nan)
        if xs:
            lbl = SHORT.get(m, m)
            line = axes[0].plot(xs, s4, marker="o", ms=3, lw=1.0, label=lbl)[0]
            c = line.get_color()
            axes[1].plot(xs, e4, marker="o", ms=3, lw=1.0, color=c)
            axes[1].plot(xs, ed, marker="s", ms=2.5, lw=0.8, ls="--", color=c,
                         alpha=0.7)
    axes[0].set_xlabel("input resolution (pixels)")
    axes[0].set_ylabel("four-thread speedup $S_4$")
    axes[0].set_title("(a) parallelism follows operator extent", fontsize=8)
    axes[0].legend(fontsize=6, framealpha=0.9)
    axes[0].axhline(1.0, color=MUT, lw=0.7, ls=":")
    axes[1].axhline(1.0, color=INK, lw=1.0, ls="--")
    axes[1].set_xlabel("input resolution (pixels)")
    axes[1].set_ylabel("$E_4/E_1$")
    axes[1].set_title("(b) energy ratio, same colours as (a)", fontsize=8)
    axes[1].plot([], [], color=MUT, marker="o", ms=3, lw=1.0,
                 label="total energy")
    axes[1].plot([], [], color=MUT, marker="s", ms=2.5, lw=0.8, ls="--",
                 label="dynamic energy")
    axes[1].legend(fontsize=6, framealpha=0.9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_resolution.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "v2_resolution.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote v2_resolution.pdf")


if __name__ == "__main__":
    fig_plane()
    fig_scaling()
    fig_perf()
    fig_resolution()
