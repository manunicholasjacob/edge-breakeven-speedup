#!/usr/bin/env python3
"""PMIC power sampling and thermal/frequency helpers for the Pi 5.

Vendored so this campaign's code directory is self-contained: the first
submission's harness imported these from a sibling paper's sweep.py, which was
not shipped in the artifact.
"""

import os
import subprocess
import threading
import time

CPUFREQ = "/sys/devices/system/cpu/cpu0/cpufreq"


class PmicSampler:
    """Samples every PMIC rail and integrates power over a window.

    The period is set to 23 ms, but a read costs more than that in practice:
    the achieved rate is about 21 Hz at one thread and 20.5 Hz at four, which
    is what the paper reports. Rails are summed into total / VDD_CORE / DDR
    groups. The DDR group is a small SoC-side rail, not the DRAM array supply.
    """

    def __init__(self, period=0.023):
        self.period = period
        self._stop = threading.Event()
        self.samples = []
        self._thr = None

    @staticmethod
    def _read():
        out = subprocess.check_output(["vcgencmd", "pmic_read_adc"]).decode()
        v, a = {}, {}
        for ln in out.strip().splitlines():
            ln = ln.strip()
            if "volt(" in ln:
                v[ln.split()[0][:-2]] = float(ln.split("=")[1].rstrip("V"))
            elif "current(" in ln:
                a[ln.split()[0][:-2]] = float(ln.split("=")[1].rstrip("A"))
        tot = core = ddr = 0.0
        for k in v:
            if k in a:
                p = v[k] * a[k]
                tot += p
                if k.startswith("VDD_CORE"):
                    core += p
                elif k.startswith("DDR"):
                    ddr += p
        return tot, core, ddr

    def _loop(self):
        while not self._stop.is_set():
            try:
                tot, core, ddr = self._read()
                self.samples.append((time.perf_counter(), tot, core, ddr))
            except Exception:
                pass
            time.sleep(self.period)

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)

    def integrate(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if len(s) < 2:
            return None
        e_tot = e_core = e_ddr = 0.0
        for i in range(1, len(s)):
            dt = s[i][0] - s[i - 1][0]
            e_tot += 0.5 * (s[i][1] + s[i - 1][1]) * dt
            e_core += 0.5 * (s[i][2] + s[i - 1][2]) * dt
            e_ddr += 0.5 * (s[i][3] + s[i - 1][3]) * dt
        span = s[-1][0] - s[0][0]
        return {
            "n_pmic_samples": len(s),
            "pmic_span_s": span,
            "energy_total_J": e_tot,
            "energy_core_J": e_core,
            "energy_ddr_J": e_ddr,
            "P_mean_W": e_tot / span if span > 0 else 0.0,
            "P_core_mean_W": e_core / span if span > 0 else 0.0,
            "P_ddr_mean_W": e_ddr / span if span > 0 else 0.0,
            "ddr_share_pct": 100.0 * e_ddr / e_tot if e_tot > 0 else 0.0,
        }


def soc_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return -1.0


def read_freqs():
    with open(os.path.join(CPUFREQ, "scaling_available_frequencies")) as f:
        return sorted(int(x) for x in f.read().split())


def limits():
    with open(os.path.join(CPUFREQ, "scaling_min_freq")) as f:
        lo = f.read().strip()
    with open(os.path.join(CPUFREQ, "scaling_max_freq")) as f:
        hi = f.read().strip()
    return lo, hi


def write_limits(lo, hi):
    for cpu in range(os.cpu_count()):
        base = "/sys/devices/system/cpu/cpu%d/cpufreq" % cpu
        if not os.path.isdir(base):
            continue
        for name, val in (("scaling_min_freq", lo), ("scaling_max_freq", hi)):
            subprocess.run(["sudo", "tee", os.path.join(base, name)],
                           input=str(val).encode(), stdout=subprocess.DEVNULL,
                           check=False)


def governor(name=None):
    path = os.path.join(CPUFREQ, "scaling_governor")
    if name is None:
        with open(path) as f:
            return f.read().strip()
    for cpu in range(os.cpu_count()):
        base = "/sys/devices/system/cpu/cpu%d/cpufreq" % cpu
        if os.path.isdir(base):
            subprocess.run(["sudo", "tee", os.path.join(base, "scaling_governor")],
                           input=name.encode(), stdout=subprocess.DEVNULL,
                           check=False)
    time.sleep(0.4)
    return governor()


def set_freq(khz):
    write_limits(khz, khz)
    time.sleep(0.4)


def cur_freq():
    try:
        with open(os.path.join(CPUFREQ, "scaling_cur_freq")) as f:
            return int(f.read())
    except Exception:
        return -1


def cooldown(target_c, max_wait_s=180.0):
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        if soc_temp_c() <= target_c:
            return True
        time.sleep(2.0)
    return False
