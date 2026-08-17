#!/usr/bin/env python3
"""Energy-optimal early exit: is the accuracy-per-joule optimum where the
latency optimum is?

Reports TOTAL board energy per inference. The excess-over-idle figure is not
usable across this sweep: at one thread the mean board power is 2.10 W against a
2.06 W idle baseline, so the dynamic component is under 2% of the signal and is
dominated by baseline drift rather than by the workload. Total energy is what a
battery actually pays for a duty-cycled device anyway.
"""

import json
import sys
from collections import defaultdict
from statistics import median


def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "energy_per_inf_mJ" in r]
    if not rows:
        print("no usable rows")
        return
    acc = {r["exit"]: r["acc_pct"] for r in rows if r.get("acc_pct")}

    cells = defaultdict(list)
    for r in rows:
        cells[(r["threads"], r["exit"], r["freq_khz"])].append(r)
    med = lambda rs, f: median([x[f] for x in rs if f in x])

    threads = sorted({k[0] for k in cells})
    exits = ["exit1", "exit2", "final"]
    freqs = sorted({k[2] for k in cells})

    print("=" * 74)
    print("1. LATENCY AND ENERGY vs CLOCK  (median of available reps)")
    print("=" * 74)
    for th in threads:
        print(f"\n-- {th} thread{'s' if th > 1 else ''} --")
        print(f"{'MHz':>6}" + "".join(f"{e+' ms':>11}" for e in exits)
              + "".join(f"{e+' mJ':>11}" for e in exits))
        for f in freqs:
            line = f"{f//1000:>6}"
            for e in exits:
                rs = cells.get((th, e, f))
                line += f"{med(rs,'lat_median_ms'):>11.3f}" if rs else f"{'-':>11}"
            for e in exits:
                rs = cells.get((th, e, f))
                line += f"{med(rs,'energy_per_inf_mJ'):>11.3f}" if rs else f"{'-':>11}"
            print(line)

    print("\n" + "=" * 74)
    print("2. WHERE IS EACH OPTIMUM?  (over the whole (threads x clock) grid)")
    print("=" * 74)
    print(f"{'exit':<8}{'fastest config':>26}{'ms':>9}{'lowest-energy config':>28}{'mJ':>9}{'energy penalty':>16}")
    for e in exits:
        pts = [(k[0], k[2], med(v, "lat_median_ms"), med(v, "energy_per_inf_mJ"))
               for k, v in cells.items() if k[1] == e]
        if not pts:
            continue
        fast = min(pts, key=lambda p: p[2])
        cheap = min(pts, key=lambda p: p[3])
        pen = 100 * (fast[3] - cheap[3]) / cheap[3]
        print(f"{e:<8}{f'{fast[0]}t @ {fast[1]//1000} MHz':>26}{fast[2]:>9.3f}"
              f"{f'{cheap[0]}t @ {cheap[1]//1000} MHz':>28}{cheap[3]:>9.3f}"
              f"{pen:>15.1f}%")

    print("\n" + "=" * 74)
    print("3. ACCURACY PER JOULE  (total energy). Which exit is most efficient?")
    print("=" * 74)
    best = []
    for th in threads:
        for e in exits:
            for f in freqs:
                rs = cells.get((th, e, f))
                if not rs:
                    continue
                mj = med(rs, "energy_per_inf_mJ")
                best.append((acc.get(e, 0) / (mj / 1000.0), e, th, f, mj,
                             med(rs, "lat_median_ms")))
    best.sort(reverse=True)
    print(f"{'rank':<5}{'exit':<8}{'thr':>4}{'MHz':>7}{'mJ/inf':>10}{'ms':>9}{'acc pts/J':>12}")
    for i, (apj, e, th, f, mj, ms) in enumerate(best[:8], 1):
        print(f"{i:<5}{e:<8}{th:>4}{f//1000:>7}{mj:>10.3f}{ms:>9.3f}{apj:>12.1f}")
    if len(best) > 8:
        print("   ...")
        for i, (apj, e, th, f, mj, ms) in enumerate(best[-3:], len(best) - 2):
            print(f"{i:<5}{e:<8}{th:>4}{f//1000:>7}{mj:>10.3f}{ms:>9.3f}{apj:>12.1f}")

    print("\n" + "=" * 74)
    print("4. THE THREAD-COUNT LEVER  (same exit, same clock, 1 vs 4 threads)")
    print("=" * 74)
    print(f"{'exit':<8}{'MHz':>7}{'ms @1t':>10}{'ms @4t':>10}{'speedup':>10}"
          f"{'mJ @1t':>10}{'mJ @4t':>10}{'energy cost':>13}")
    for e in exits:
        for f in freqs:
            a, b = cells.get((1, e, f)), cells.get((4, e, f))
            if not a or not b:
                continue
            la, lb = med(a, "lat_median_ms"), med(b, "lat_median_ms")
            ea, eb = med(a, "energy_per_inf_mJ"), med(b, "energy_per_inf_mJ")
            print(f"{e:<8}{f//1000:>7}{la:>10.3f}{lb:>10.3f}{la/lb:>9.2f}x"
                  f"{ea:>10.3f}{eb:>10.3f}{eb/ea:>12.2f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/P2_exit_energy.jsonl")
