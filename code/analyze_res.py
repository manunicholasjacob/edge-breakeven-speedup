#!/usr/bin/env python3
"""Is the resolution crossover architecture-dependent, or a property of the board?

The letter establishes that input resolution sets the parallel speedup using two
architectures, and its limitations section concedes that whether the crossover
resolution itself varies by architecture is open. Three more architectures answer
it. The quantity of interest is the resolution at which E_4/E_1 crosses 1, found by
linear interpolation between the bracketing measurements.
"""

import collections
import json
import statistics
import sys


def load(paths):
    agg = collections.defaultdict(lambda: {"ms": [], "mJ": []})
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if "error" in r or "ms_per_inf" not in r:
                continue
            k = (r["model"].replace(".onnx", ""), r["resolution"], r["threads"])
            agg[k]["ms"].append(r["ms_per_inf"])
            agg[k]["mJ"].append(r.get("mJ_per_inf", 0.0))
    return {k: {"ms": statistics.median(v["ms"]), "mJ": statistics.median(v["mJ"])}
            for k, v in agg.items()}


def crossing(points):
    """Resolution where the energy ratio crosses 1, by linear interpolation."""
    pts = sorted(points)
    for (r0, e0), (r1, e1) in zip(pts, pts[1:]):
        if (e0 - 1.0) * (e1 - 1.0) <= 0 and e0 != e1:
            return r0 + (r1 - r0) * (e0 - 1.0) / (e0 - e1)
    return None


def main(paths):
    a = load(paths)
    models = sorted(set(m for m, _, _ in a))
    res = sorted(set(r for _, r, _ in a))

    print(f"{'architecture':<22}" + "".join(f"{r:>16}" for r in res))
    print(f"{'':<22}" + "".join(f"{'S4    E4/E1':>16}" for _ in res))
    cross = {}
    for m in models:
        cells, epts = [], []
        for r in res:
            b, c = a.get((m, r, 1)), a.get((m, r, 4))
            if not b or not c:
                cells.append("-")
                continue
            S = b["ms"] / c["ms"]
            E = (c["mJ"] / b["mJ"]) if b["mJ"] else float("nan")
            epts.append((r, E))
            cells.append(f"{S:.2f}  {E:.2f}")
        print(f"{m:<22}" + "".join(f"{c:>16}" for c in cells))
        cross[m] = crossing(epts)

    print("\nresolution at which four threads start saving energy (E_4/E_1 = 1):")
    vals = []
    for m in models:
        c = cross[m]
        print(f"  {m:<22}{'never in range' if c is None else f'{c:6.0f} px'}")
        if c is not None:
            vals.append(c)
    if len(vals) > 1:
        print(f"\n  spread across architectures: {min(vals):.0f} to {max(vals):.0f} px "
              f"(median {statistics.median(vals):.0f})")

    print("\nspeedup at the extremes (four threads):")
    print(f"{'architecture':<22}{'S4 @32':>10}{'S4 @224':>10}{'ratio':>8}")
    for m in models:
        b32, c32 = a.get((m, 32, 1)), a.get((m, 32, 4))
        b224, c224 = a.get((m, 224, 1)), a.get((m, 224, 4))
        if not (b32 and c32 and b224 and c224):
            continue
        s32, s224 = b32["ms"] / c32["ms"], b224["ms"] / c224["ms"]
        print(f"{m:<22}{s32:>10.2f}{s224:>10.2f}{s224/s32:>8.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
