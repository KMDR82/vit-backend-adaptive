# Cell 11 — contention grid (Ada):
# 4 duty levels {0,25,50,100} x 7 backends x 4 models x batch {1,8} x res 224
# x 3 seeds. Reuses cell8_grid's builders. Calibrates each level (30 s NVML
# util+power, GPU otherwise idle) before measuring. Resume-safe CSV.
import os, sys, subprocess, time, csv, random, threading
from pathlib import Path
import numpy as np

ROOT = Path.home() / "projects" / "vit_backend_adaptive"
sys.path.insert(0, str(ROOT))
import cell8_grid as g8          # lib preload + CC happen at import
import torch, pynvml

HW_ID = g8.HW_ID
PY = sys.executable
LEVELS = [0, 25, 50, 100]
BATCHES = [1, 8]; RES = 224
N_TIMED = 100                     # shorter latency window for the slice
SEEDS = [42, 1234, 2024]
H = pynvml.nvmlDeviceGetHandleByIndex(0)   # g8 already did nvmlInit

def start_contender(duty):
    if duty == 0:
        return None
    p = subprocess.Popen([PY, str(ROOT / "contender.py"), str(duty)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5.0)               # ramp
    return p

def stop_contender(p):
    if p is not None:
        p.terminate(); p.wait(timeout=10)
    torch.cuda.synchronize(); time.sleep(2.0)

def calibrate():
    """30 s of NVML util+power per level, GPU otherwise idle. Once per run."""
    out = ROOT / "output" / f"contention_calibration_{HW_ID}.csv"
    if out.exists():
        print("calibration exists, skipping"); return
    rows = [("duty", "util_mean", "util_p95", "power_mean_W", "sm_clock_mean")]
    for duty in LEVELS:
        p = start_contender(duty)
        us, ps, cs = [], [], []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 30:
            us.append(pynvml.nvmlDeviceGetUtilizationRates(H).gpu)
            ps.append(pynvml.nvmlDeviceGetPowerUsage(H) / 1000.0)
            cs.append(pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM))
            time.sleep(0.1)
        stop_contender(p)
        rows.append((duty, f"{np.mean(us):.1f}", f"{np.percentile(us,95):.1f}",
                     f"{np.mean(ps):.1f}", f"{np.mean(cs):.0f}"))
        print(f"duty {duty:3d}%: util {np.mean(us):5.1f}%  "
              f"power {np.mean(ps):6.1f} W  clock {np.mean(cs):.0f} MHz", flush=True)
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)

OUT = ROOT / "output" / f"contention_grid_{HW_ID}.csv"
if not OUT.exists():
    OUT.write_text("seed,contention,backend,model,batch,res,status,ep,"
        "lat_med_ms,lat_p5_ms,lat_p95_ms,lat_mean_ms,"
        "energy_mJ_per_inf,power_mean_W,n_power_updates,energy_win_s,"
        "mem_MB,idle_mem_MB,sm_clock_mean,sm_clock_min,"
        "miss30,miss60,miss120,setup_s\n")

def done_keys():
    d = set()
    with open(OUT) as f:
        for r in csv.DictReader(f):
            d.add((r["seed"], r["contention"], r["backend"], r["model"], r["batch"]))
    return d

def main():
    calibrate()
    g8.N_TIMED = N_TIMED          # shorten latency window for the slice
    done = done_keys()
    grid = [(be, mn, b) for be in g8.BACKENDS for mn in g8.MODELS for b in BATCHES]
    for duty in LEVELS:
        p = start_contender(duty)
        try:
            for seed in SEEDS:
                order = grid[:]; random.Random(seed).shuffle(order)
                for be, mn, b in order:
                    key = (str(seed), str(duty), be, mn, str(b))
                    if key in done: continue
                    tag = f"c{duty} s{seed} {be} {mn} b{b}"
                    try:
                        r = g8.one_run(seed, be, mn, b, RES)
                        row = [seed, duty, be, mn, b, RES, r["status"], r["ep"]] + [
                            f"{r[k]:.4f}" for k in
                            ["lat_med_ms","lat_p5_ms","lat_p95_ms","lat_mean_ms",
                             "energy_mJ_per_inf","power_mean_W","n_power_updates",
                             "energy_win_s","mem_MB","idle_mem_MB","sm_clock_mean",
                             "sm_clock_min","miss30","miss60","miss120","setup_s"]]
                        print(f"[{time.strftime('%H:%M:%S')}] {tag}: "
                              f"lat={r['lat_med_ms']:.2f}ms E={r['energy_mJ_per_inf']:.2f}mJ",
                              flush=True)
                    except torch.cuda.OutOfMemoryError:
                        row = [seed, duty, be, mn, b, RES, "OOM", ""] + [""] * 16
                        print(f"[{time.strftime('%H:%M:%S')}] {tag}: OOM", flush=True)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        row = [seed, duty, be, mn, b, RES, "ERROR", str(e)[:100]] + [""] * 16
                        print(f"[{time.strftime('%H:%M:%S')}] {tag}: ERROR {str(e)[:100]}",
                              flush=True)
                    with open(OUT, "a") as f:
                        csv.writer(f).writerow(row)
                    torch.cuda.empty_cache()
        finally:
            stop_contender(p)
    print("Cell 11 complete.", flush=True)

if __name__ == "__main__":
    main()
