# Cell 17 — reviewer analyses on existing data (no GPU needed).
# (A) Resident engines: replay S1/S2 (and S3 on the Ada GPU) with switching cost
#     S = 0, i.e. all engines kept loaded in GPU memory. Same traces, seeds,
#     lambda values and selector parameters as cell13_replay.py / cell14_s3.py.
# (B) Tail latency and deadline misses: per-bucket p99 latency from the raw
#     traces and miss rates at 30/60/120 fps, for the static grid and, on the
#     Ada GPU, for the saturated-contention level.
# Usage:  HW=RTX_4500_Ada_Generation python cell17_resident_tail.py
#         HW=Tesla_T4               python cell17_resident_tail.py
# Outputs: output/replay_results_resident_<HW>.csv, output/tail_<HW>.csv,
#          output/cell17_summary_<HW>.txt
import os, sys, csv, itertools
from pathlib import Path
import numpy as np

os.environ.setdefault("HW", "RTX_4500_Ada_Generation")
ROOT = Path.home() / "projects" / "vit_backend_adaptive"
sys.path.insert(0, str(ROOT))
import cell13_replay as c13
from bandit_selectors import SWUCB, DUCB, EXP3S, GLRklUCB, EpsGreedy, RandomSelector

HW = c13.HW_ID
OUT = ROOT / "output"
IS_ADA = HW.startswith("RTX_4500")
MODELS = ["vit_tiny_patch16_224", "vit_small_patch16_224",
          "vit_base_patch16_224", "deit_small_patch16_224"]
SEEDS = [42, 1234, 2024]
SELECTORS = ["GLR-klUCB", "D-UCB", "SW-UCB", "eps", "EXP3.S", "random"]
LOG = []

def say(s=""):
    print(s, flush=True); LOG.append(s)

def make_selectors(K, env, seed):
    Z = np.zeros((K, K)); scale = float(np.median(env))
    return {"random": RandomSelector(K, switch_costs=Z, rng=np.random.default_rng(seed)),
            "eps": EpsGreedy(K, switch_costs=Z, rng=np.random.default_rng(seed)),
            "SW-UCB": SWUCB(K, tau=300, switch_costs=Z),
            "D-UCB": DUCB(K, gamma=0.995, switch_costs=Z),
            "EXP3.S": EXP3S(K, cost_scale=2 * scale, switch_costs=Z,
                            rng=np.random.default_rng(seed)),
            "GLR-klUCB": GLRklUCB(K, switch_costs=Z)}

# ---------------------------------------------------------------- (A) resident
def part_a():
    say(f"== (A) resident engines, S = 0  [{HW}]")
    grid = c13.load_grid()
    runs = []
    for kind, model, seed in itertools.product(["S1", "S2"], MODELS, SEEDS):
        env, S_list, pid, arms, _ = c13.build_scenario(grid, seed, model, kind)
        runs.append((kind, model, seed, env, pid, arms))
    if IS_ADA:
        import cell14_s3 as c14
        lam_E, _, _ = c13.fix_lambda_E(grid)
        rows = {"c0": c14.read_rows(c14.RERUN), "c100": c14.read_rows(c14.CONT, contention="100")}
        S_c0 = c13.load_switch_costs()
        for b, model, seed in itertools.product([1, 8], MODELS, SEEDS):
            env, S_list, pid, arms = c14.build_s3(seed, model, b, lam_E, rows, S_c0)
            runs.append((f"S3b{b}", model, seed, env, pid, arms))
    out = OUT / f"replay_results_resident_{HW}.csv"
    res = {}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "model", "seed", "oracle_arms", "C_static", "C_oracle",
                    "selector", "C_T", "switches", "delta_vs_static"])
        for kind, model, seed, env, pid, arms in runs:
            K = env.shape[1]
            S0 = [np.zeros((K, K)) for _ in np.unique(pid)]
            a_star, C_st, C_or, oarms = c13.baselines_dynamic(env, S0, pid)
            grp = "S3" if kind.startswith("S3") else "S1S2"
            res.setdefault((grp, "oracle"), []).append(C_st - C_or)
            for name, sel in make_selectors(K, env, seed).items():
                C_T, sw = c13.replay_dynamic(env, S0, pid, sel)
                w.writerow([kind, model, seed, "|".join(arms[a] for a in oarms),
                            f"{C_st:.0f}", f"{C_or:.0f}", name, f"{C_T:.0f}", sw,
                            f"{C_st - C_T:.0f}"])
                res.setdefault((grp, name), []).append(C_st - C_T)
                res.setdefault((grp, name, "sw"), []).append(sw)
    say(f"saved {out}")
    say("median delta vs static (positive = beats best-static), runs with delta>0, median switches")
    for grp in ["S1S2", "S3"]:
        if (grp, "oracle") not in res:
            continue
        o = np.array(res[(grp, "oracle")])
        say(f"  [{grp}] n={len(o)}  oracle: median {np.median(o):.0f}, "
            f"max {o.max():.0f}, runs>0: {(o > 0).sum()}")
        for n in SELECTORS:
            d = np.array(res[(grp, n)])
            say(f"  [{grp}] {n:10s} median {np.median(d):>10.0f}  runs>0: {(d > 0).sum():2d}/{len(d)}"
                f"  median sw {np.median(res[(grp, n, 'sw')]):.0f}")

# ---------------------------------------------------------------- (B) tails
def tail_rows(trace_dir, cfgs, label):
    rows = []
    for (be, mn, b, r) in cfgs:
        p99, m30, m60, m120, med = [], [], [], [], []
        for s in SEEDS:
            p = trace_dir / f"{s}_{be}_{mn}_{b}_{r}.npz"
            if not p.exists():
                continue
            lat = np.load(p)["lats"]
            med.append(np.median(lat)); p99.append(np.percentile(lat, 99))
            m30.append(np.mean(lat > 1000 / 30)); m60.append(np.mean(lat > 1000 / 60))
            m120.append(np.mean(lat > 1000 / 120))
        if med:
            rows.append([label, be, mn, b, r, np.median(med), np.median(p99),
                         np.median(m30), np.median(m60), np.median(m120)])
    return rows

def part_b():
    say(f"\n== (B) tail latency and deadline misses  [{HW}]")
    cfgs = [(be, mn, b, r) for be in c13.BACKENDS for mn in MODELS
            for b in [1, 8, 32] for r in [224, 384]]
    rows = tail_rows(c13.RAW, cfgs, "static")
    if IS_ADA:
        c100 = OUT / f"contention_raw_{HW}" / "c100"
        rows += tail_rows(c100, [(be, mn, b, 224) for be in c13.BACKENDS
                                 for mn in MODELS for b in [1, 8]], "c100")
    out = OUT / f"tail_{HW}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "backend", "model", "batch", "res", "lat_med_ms",
                    "lat_p99_ms", "miss30", "miss60", "miss120"])
        for r in rows:
            w.writerow(r[:5] + [f"{x:.4f}" for x in r[5:]])
    say(f"saved {out}")
    for level in ["static", "c100"]:
        R = [r for r in rows if r[0] == level]
        if not R:
            continue
        buckets = sorted({(r[2], r[3], r[4]) for r in R})
        same, diffs, trt_tail = 0, [], []
        for bk in buckets:
            B = [r for r in R if (r[2], r[3], r[4]) == bk]
            w_med = min(B, key=lambda r: r[5]); w_p99 = min(B, key=lambda r: r[6])
            same += w_med[1] == w_p99[1]
            if w_med[1] != w_p99[1]:
                diffs.append(f"{bk}: median-winner {w_med[1]} p99-winner {w_p99[1]} "
                             f"({w_med[6]:.2f} vs {w_p99[6]:.2f} ms)")
            t = [r for r in B if r[1] == "ort_trt"]
            if t:
                trt_tail.append(t[0][6] / t[0][5])
        say(f"  [{level}] buckets {len(buckets)}: latency winner identical by median and p99 "
            f"in {same}; TRT p99/median ratio median {np.median(trt_tail):.2f}, max {np.max(trt_tail):.2f}")
        for d in diffs:
            say(f"    differs: {d}")
        for f_, col in [(30, 7), (60, 8), (120, 9)]:
            trt = [r[col] for r in R if r[1] == "ort_trt"]
            allb = [r[col] for r in R]
            say(f"  [{level}] miss@{f_}fps: TRT buckets with any miss {sum(x > 0 for x in trt)}/{len(trt)} "
                f"(max {max(trt):.3f}); all arms max {max(allb):.3f}")

if __name__ == "__main__":
    part_a()
    part_b()
    (OUT / f"cell17_summary_{HW}.txt").write_text("\n".join(LOG) + "\n")
    print(f"summary written to output/cell17_summary_{HW}.txt")
