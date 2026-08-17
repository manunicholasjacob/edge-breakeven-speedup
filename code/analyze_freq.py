#!/usr/bin/env python3
"""Does the break-even rule hold at every clock, or only at the one we reported?

The letter validates S_n > P_n/P_1 at 2400 MHz. The dataset covers four clocks,
and the clock is exactly the knob that moves the static-to-dynamic power split,
which is what sets the power ratio. So the unused three quarters of the data are a
test of the formulation rather than a repetition of it: if the break-even threshold
is a real quantity it should move with the operating point, and the rule should keep
predicting the sign of the energy effect after it moves.

Reports, per frequency: the spread of the power ratio across models (is it still a
board constant at that clock?), the break-even threshold it implies, and the number
of model-by-thread cases where sign(S_n - P_n/P_1) agrees with sign(1 - E_n/E_1).
"""

import collections
import json
import statistics
import sys


def load(path):
    rows = [json.loads(l) for l in open(path)]
    # median over repetitions for each (model, threads, freq)
    agg = collections.defaultdict(lambda: {"lat": [], "e": [], "p": []})
    for r in rows:
        if "lat_median_ms" not in r or "energy_per_inf_mJ" not in r:
            continue
        k = (r["model"], r["threads"], r["freq_khz"])
        agg[k]["lat"].append(r["lat_median_ms"])
        agg[k]["e"].append(r["energy_per_inf_mJ"])
        agg[k]["p"].append(r["P_mean_W"])
    return {k: {kk: statistics.median(v) for kk, v in d.items()} for k, d in agg.items()}


def main(path):
    a = load(path)
    models = sorted(set(m for m, _, _ in a))
    freqs = sorted(set(f for _, _, f in a))

    print(f"{'MHz':>6}{'n':>4}{'P_n/P_1 range':>18}{'median':>9}"
          f"{'S_n range':>16}{'sign correct':>14}")
    grand_ok = grand_n = 0
    per_freq = {}
    for f in freqs:
        for n in (2, 4):
            pr, sp, rows = [], [], []
            for m in models:
                b, c = a.get((m, 1, f)), a.get((m, n, f))
                if not b or not c:
                    continue
                P = c["p"] / b["p"]
                S = b["lat"] / c["lat"]
                E = c["e"] / b["e"]
                pr.append(P)
                sp.append(S)
                rows.append((m, S, P, E))
            if not rows:
                continue
            ok = sum(1 for _, S, P, E in rows if (S > P) == (E < 1.0))
            grand_ok += ok
            grand_n += len(rows)
            per_freq[(f, n)] = (rows, pr, sp, ok)
            print(f"{f//1000:>6}{n:>4}{min(pr):>9.2f}-{max(pr):<8.2f}"
                  f"{statistics.median(pr):>9.2f}"
                  f"{min(sp):>8.2f}-{max(sp):<7.2f}{ok:>8d}/{len(rows):<5d}")

    print(f"\nsign predicted correctly in {grand_ok} of {grand_n} model-by-thread-by-clock cases")

    print("\nbreak-even threshold by clock (median power ratio at 4 threads):")
    for f in freqs:
        if (f, 4) in per_freq:
            _, pr, _, _ = per_freq[(f, 4)]
            print(f"  {f//1000:>5} MHz   {statistics.median(pr):.2f}x"
                  f"   (spread {min(pr):.2f}-{max(pr):.2f})")

    print("\nper-model detail at 4 threads, every clock "
          "(S = speedup, P = power ratio, E = energy ratio):")
    print(f"{'model':<22}" + "".join(f"{f//1000:>17}" for f in freqs))
    for m in models:
        cells = []
        for f in freqs:
            rows = per_freq.get((f, 4), ([],))[0]
            r = [x for x in rows if x[0] == m]
            cells.append(f"S{r[0][1]:.2f} E{r[0][3]:.2f}" if r else "-")
        print(f"{m:<22}" + "".join(f"{c:>17}" for c in cells))

    # the interesting question: does any model change sign across the clock range?
    print("\nmodels whose energy verdict changes with clock, 4 threads:")
    flips = 0
    for m in models:
        verdicts = []
        for f in freqs:
            rows = per_freq.get((f, 4), ([],))[0]
            r = [x for x in rows if x[0] == m]
            if r:
                verdicts.append(r[0][3] < 1.0)
        if verdicts and len(set(verdicts)) > 1:
            flips += 1
            print(f"  {m}: {['save' if v else 'cost' for v in verdicts]}")
    if not flips:
        print("  none: every model keeps its verdict across the whole 1500-2400 MHz range")


if __name__ == "__main__":
    main(sys.argv[1])
