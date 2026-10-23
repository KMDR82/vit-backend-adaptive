# Cell 14 — S3 contention scenario replay (Ada). Reuses cell13_replay's lambda_E,
# cost model, trace resampling and the identical selector set/parameters, so
# S1/S2/S3 come from the same code path.
#
# Phases: c0 -> c100 -> c0 -> c100 -> c0 (5 x 300 steps, as S1/S2), r224,
# one scenario per batch size: S3b1 (batch 1) and S3b8 (batch 8);
# 4 models x 3 seeds x 2 = 24 runs, same count as S1/S2.
# Data sources (documented in the paper):
#   c0  latency traces : output/static_grid_raw_<HW>/ (refilled by rerun224.py)
#   c0  energy         : output/static_grid_rerun224_<HW>.csv (same run as traces)
#   c0  switch costs   : cell13.load_switch_costs() = original static grid, max
#                        setup_s over seeds (same rule as S1/S2)
#   c100 traces        : output/contention_raw_<HW>/c100/
#   c100 energy, S     : output/contention_grid_<HW>.csv rows with contention=100,
#                        S = max setup_s over seeds measured under contention
# Output: output/replay_results_S3_<HW>.csv (same columns as replay_results_<HW>.csv)
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
RERUN = OUT / f"static_grid_rerun224_{HW}.csv"
CONT = OUT / f"contention_grid_{HW}.csv"
C100_RAW = OUT / f"contention_raw_{HW}" / "c100"
PHASES = ["c0", "c100", "c0", "c100", "c0"]
MODELS = ["vit_tiny_patch16_224", "vit_small_patch16_224",
          "vit_base_patch16_224", "deit_small_patch16_224"]
K_ALL = len(c13.BACKENDS)
SHORT = {"vit_tiny_patch16_224": "ViT-T", "vit_small_patch16_224": "ViT-S",
         "vit_base_patch16_224": "ViT-B", "deit_small_patch16_224": "DeiT-S"}

def read_rows(path, contention=None):
    rows = {}
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok": continue
        if contention is not None and r.get("contention") != contention: continue
        key = (int(r["seed"]), r["backend"], r["model"], int(r["batch"]), int(r["res"]))
        rows[key] = {"energy": float(r["energy_mJ_per_inf"]), "setup_s": float(r["setup_s"])}
    return rows

def s_matrix(rows, model, b, res=224):
    mat = np.zeros((K_ALL, K_ALL))
    for j, be in enumerate(c13.BACKENDS):
        v = [x["setup_s"] for k, x in rows.items() if k[1:] == (be, model, b, res)]
        if v: mat[:, j] = max(v) * 1000.0
    np.fill_diagonal(mat, 0.0)
    return mat

def trace(level, seed, be, mn, b, res=224):
    p = (c13.RAW if level == "c0" else C100_RAW) / f"{seed}_{be}_{mn}_{b}_{res}.npz"
    return np.load(p)["lats"] if p.exists() else None

def build_s3(seed, model, b, lam_E, rows, S_c0):
    rng = np.random.default_rng(seed)
    T = c13.PHASE_LEN * len(PHASES)
    env = np.full((T, K_ALL), np.nan); pid = np.zeros(T, int); S_list = []
    for i, lvl in enumerate(PHASES):
        sl = slice(i * c13.PHASE_LEN, (i + 1) * c13.PHASE_LEN); pid[sl] = i
        S_list.append(S_c0[(model, b, 224)] if lvl == "c0" else s_matrix(rows["c100"], model, b))
        for j, be in enumerate(c13.BACKENDS):
            key = (seed, be, model, b, 224)
            tr = trace(lvl, *key)
            if tr is None or key not in rows[lvl]: continue
            lat = rng.choice(tr, size=c13.PHASE_LEN, replace=True)
            miss = (lat > 1000.0 / c13.F_TARGET).astype(float)
            env[sl, j] = lat + lam_E * rows[lvl][key]["energy"] + c13.LAMBDA_D * miss
    ok = ~np.isnan(env).any(axis=0)
    return env[:, ok], [S[np.ix_(ok, ok)] for S in S_list], pid, \
           [c13.BACKENDS[j] for j in range(K_ALL) if ok[j]]

def main():
    lam_E, Lref, Eref = c13.fix_lambda_E(c13.load_grid())
    rows = {"c0": read_rows(RERUN), "c100": read_rows(CONT, contention="100")}
    S_c0 = c13.load_switch_costs()
    n_c100 = len(list(C100_RAW.glob("*.npz")))
    print(f"lambda_E = {lam_E:.4f} | c0 rows {len(rows['c0'])} | c100 rows {len(rows['c100'])} "
          f"| c100 traces {n_c100}", flush=True)
    if len(rows["c0"]) != 168 or len(rows["c100"]) != 168 or n_c100 != 168:
        sys.exit("input check failed: expected 168/168/168")
    out = OUT / f"replay_results_S3_{HW}.csv"
    summary = {}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario","model","seed","arms","oracle_arms","inversion",
                    "C_static","C_oracle","selector","C_T","switches","delta_vs_static"])
        for b, model, seed in itertools.product([1, 8], MODELS, [42, 1234, 2024]):
            kind = f"S3b{b}"
            env, S_list, pid, arms = build_s3(seed, model, b, lam_E, rows, S_c0)
            K = env.shape[1]
            Sbar = np.mean(S_list, axis=0); scale = float(np.median(env))
            a_star, C_st, C_or, oarms = c13.baselines_dynamic(env, S_list, pid)
            inv = len(set(oarms)) > 1
            mk = {"random": RandomSelector(K, switch_costs=Sbar, rng=np.random.default_rng(seed)),
                  "eps": EpsGreedy(K, switch_costs=Sbar, rng=np.random.default_rng(seed)),
                  "SW-UCB": SWUCB(K, tau=300, switch_costs=Sbar),
                  "D-UCB": DUCB(K, gamma=0.995, switch_costs=Sbar),
                  "EXP3.S": EXP3S(K, cost_scale=2*scale, switch_costs=Sbar,
                                  rng=np.random.default_rng(seed)),
                  "GLR-klUCB": GLRklUCB(K, switch_costs=Sbar)}
            for name, sel in mk.items():
                C_T, sw = c13.replay_dynamic(env, S_list, pid, sel)
                w.writerow([kind, model, seed, "|".join(arms),
                            "|".join(arms[a] for a in oarms), int(inv),
                            f"{C_st:.0f}", f"{C_or:.0f}", name, f"{C_T:.0f}", sw,
                            f"{C_st - C_T:.0f}"])
                summary.setdefault((kind, name), []).append(C_st - C_T)
            summary.setdefault((kind, "oracle"), []).append(C_st - C_or)
            print(f"{kind} {SHORT[model]:6s} s{seed}: inversion={inv} "
                  f"a*={arms[a_star]} oracle={'>'.join(arms[a] for a in oarms)} "
                  f"oracle_delta={C_st - C_or:.0f}", flush=True)
    print(f"saved {out}")
    print("\nmedian delta vs static (positive = beats best-static):")
    for kind in ["S3b1", "S3b8"]:
        print("  " + kind + ": " + ", ".join(
            f"{n}={np.median(summary[(kind, n)]):.0f}"
            for n in ["oracle", "GLR-klUCB", "D-UCB", "SW-UCB", "eps", "EXP3.S", "random"]))

if __name__ == "__main__":
    main()
