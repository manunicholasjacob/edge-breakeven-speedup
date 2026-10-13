#!/usr/bin/env python3
"""Fail if the README's headline numbers no longer follow from data/v2/.

Runs code/v2/fill_numbers_v2.py, which regenerates every manuscript number from the released
data into paper-v2/numbers_v2.tex, then compares the macros behind each README claim with the
value the README states. Standard library only.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "paper-v2", "numbers_v2.tex")

# macro -> (value the README states, README claim it backs)
EXPECTED = {
    "npNModels": ("8", "eight ImageNet-scale networks"),
    "npBreakTotalTwentyFour": ("1.82", "break-even when charging the floor is 1.82x"),
    "npSaveRangeTotal": (r"12--27\%", "charging the floor saves 12-27% of board energy"),
    "npNFlip": ("8", "all eight save with the floor charged and lose without it"),
    "npNClearPinned": ("0", "at four threads none of the eight clears"),
    "npCiReps": ("10", "ten repetitions per cell"),
    "npCiNModels": ("8", "bootstrap intervals for all eight"),
    "npCiNAbove": ("8", "whole 95% interval above break-even for all eight"),
    "npBreakFloorLo": ("0.82", "flip floor 0.82 W (low end)"),
    "npBreakFloorHi": ("1.46", "flip floor 1.46 W (high end)"),
    "npFloorAcrossLo": ("1.88", "measured floor 1.88 W (low end)"),
    "npFloorAcrossHi": ("2.03", "measured floor 2.03 W (high end)"),
    "npIdleFifteen": ("1.72", "static floor 1.72 W at 1.5 GHz"),
    "npIdleTwentyFour": ("2.00", "static floor 2.00 W at 2.4 GHz"),
    "npDynShareOne": ("44", "dynamic term 44% of the signal (1 thread)"),
    "npDynShareFour": ("69", "dynamic term 69% of the signal (4 threads)"),
    "npSpinShareBig": ("94", "disabling spin-wait removes 94% of the instruction growth"),
}


def main():
    subprocess.run([sys.executable, os.path.join(HERE, "fill_numbers_v2.py")], check=True)
    with open(OUT) as f:
        text = f.read()
    macros = dict(re.findall(r"^.newcommand[{].(np[A-Za-z]+)[}][{](.*)[}]$", text, re.M))
    tbd = "TBD" in text
    failed = []
    for name, (want, claim) in EXPECTED.items():
        got = macros.get(name)
        ok = got == want
        print(f"{'ok  ' if ok else 'FAIL'} {name} = {got!r} (README: {want}; {claim})")
        if not ok:
            failed.append(name)
    if tbd:
        print("FAIL the generated numbers file still contains TBD placeholders")
    if failed or tbd:
        sys.exit(1)
    print(f"\nall {len(EXPECTED)} headline numbers reproduce from data/v2/ (no macro left TBD)")


if __name__ == "__main__":
    main()
