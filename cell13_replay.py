# Cell 13 — replay of scenarios S1 (batch) and S2 (resolution).
# Part 1 loads the static grid, raw traces and switch costs; part 2 builds the
# scenarios, checks for rank inversions and replays the selectors.
# Usage: HW=<gpu id> python cell13_replay.py run
import csv, sys
from pathlib import Path
import numpy as np

ROOT = Path.home() / "projects" / "vit_backend_adaptive"
import os as _os
HW_ID = _os.environ.get("HW", "RTX_4500_Ada_Generation")
GRID = ROOT / "output" / f"static_grid_{HW_ID}.csv"
RAW  = ROOT / "output" / f"static_grid_raw_{HW_ID}"
SETUP = ROOT / "output" / f"setup_costs_{HW_ID}.csv"

BACKENDS = ["eager","torchscript","inductor_default","inductor_maxautotune",
            "ort_cuda_fp32","ort_cuda_fp16","ort_trt"]

def load_grid():
    """rows[(seed,backend,model,batch,res)] = dict of summary metrics (ok only)."""
    rows = {}
    with open(GRID) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            key = (int(r["seed"]), r["backend"], r["model"], int(r["batch"]), int(r["res"]))
            rows[key] = {k: float(r[k]) for k in
                         ["lat_med_ms","energy_mJ_per_inf","mem_MB",
                          "miss30","miss60","miss120","setup_s"]}
    return rows

def load_trace(seed, be, mn, b, res):
    """raw per-iteration latencies (ms) for one config."""
    p = RAW / f"{seed}_{be}_{mn}_{b}_{res}.npz"
    return np.load(p)["lats"] if p.exists() else None

def load_switch_costs():
    """S[a->a'] proxy: cold total setup time of the TARGET backend per
    (model,batch,res), taken from the Cell 8 grid itself: max setup_s over
    seeds (max ~ cold; later seeds hit warm caches). Charged on entering."""
    totals = {}   # (mn,b,res,backend) -> max setup_s
    with open(GRID) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            k = (r["model"], int(r["batch"]), int(r["res"]), r["backend"])
            v = float(r["setup_s"])
            totals[k] = max(totals.get(k, 0.0), v)
    S = {}
    keys = {(m, b, r) for (m, b, r, _) in totals}
    K = len(BACKENDS)
    import numpy as _np
    for (m, b, r) in keys:
        mat = _np.zeros((K, K))
        for j, tgt in enumerate(BACKENDS):
            t = totals.get((m, b, r, tgt))
            if t is not None:
                mat[:, j] = t * 1000.0
        _np.fill_diagonal(mat, 0.0)
        S[(m, b, r)] = mat
    return S

if __name__ == "__main__":
    grid = load_grid()
    print(f"grid rows (ok): {len(grid)}")
    seeds = sorted({k[0] for k in grid}); print(f"seeds: {seeds}")
    # coverage per backend
    for be in BACKENDS:
        n = sum(1 for k in grid if k[1] == be)
        print(f"  {be:22s} {n:3d} configs")
    # one sample trace
    if grid:
        k0 = next(iter(grid))
        tr = load_trace(*k0)
        print(f"sample trace {k0}: n={None if tr is None else len(tr)}, "
              f"med={None if tr is None else round(float(np.median(tr)),3)} ms")
    S = load_switch_costs()
    print(f"switch-cost matrices: {len(S)} (model,batch,res) buckets")
    if S:
        kk = next(iter(S))
        print(f"sample S{kk} row0 (ms): {np.round(S[kk][0], 0)}")

# ================= Part 2: scenarios, inversion check, selector replay =================
# Phases: 5 x 300 steps. lambda_E fixed so that lambda_E*med(E)
# matches med(L) on the reference config (vit_small, b8, r224, across backends);
# lambda_D = one 60fps frame budget (1000/60 ms).
import itertools
sys.path.insert(0, str(ROOT))
from bandit_selectors import (SWUCB, DUCB, EXP3S, GLRklUCB, EpsGreedy,
                              RandomSelector)

PHASE_LEN = 300
F_TARGET = 60
LAMBDA_D = 1000.0 / F_TARGET

def fix_lambda_E(grid):
    ref = [(v["lat_med_ms"], v["energy_mJ_per_inf"]) for k, v in grid.items()
           if k[2] == "vit_small_patch16_224" and k[3] == 8 and k[4] == 224]
    L = np.median([x[0] for x in ref]); E = np.median([x[1] for x in ref])
    return float(L / E), float(L), float(E)

def build_scenario(grid, seed, model, kind):
    """Returns env_costs (T,K), per-phase S matrices list, phase_ids, phase spec."""
    if kind == "S1":
        phases = [(b, 224) for b in [1, 32, 8, 1, 32]]
    elif kind == "S2":
        phases = [(8, r) for r in [224, 384, 224, 384, 224]]
    else:
        raise ValueError(kind)
    lam_E, _, _ = fix_lambda_E(grid)
    rng = np.random.default_rng(seed)
    K = len(BACKENDS)
    T = PHASE_LEN * len(phases)
    env = np.full((T, K), np.nan)
    phase_ids = np.zeros(T, int)
    S_list = []
    Sc = load_switch_costs()
    for i, (b, r) in enumerate(phases):
        sl = slice(i * PHASE_LEN, (i + 1) * PHASE_LEN)
        phase_ids[sl] = i
        S_list.append(Sc.get((model, b, r), np.zeros((K, K))))
        for j, be in enumerate(BACKENDS):
            key = (seed, be, model, b, r)
            tr = load_trace(*key)
            if tr is None or key not in grid:
                continue  # stays NaN -> config excluded below
            lat = rng.choice(tr, size=PHASE_LEN, replace=True)
            e = grid[key]["energy_mJ_per_inf"]
            miss = (lat > 1000.0 / F_TARGET).astype(float)
            env[sl, j] = lat + lam_E * e + LAMBDA_D * miss
    ok = ~np.isnan(env).any(axis=0)
    return env[:, ok], [S[np.ix_(ok, ok)] for S in S_list], phase_ids, \
           [BACKENDS[j] for j in range(K) if ok[j]], phases

def replay_dynamic(env, S_list, phase_ids, selector):
    T, K = env.shape
    total, switches, prev = 0.0, 0, None
    for t in range(T):
        a = selector.select()
        c = float(env[t, a]); selector.observe(a, c); total += c
        if prev is not None and a != prev:
            total += float(S_list[phase_ids[t]][prev, a]); switches += 1
        prev = a
    return total, switches

def baselines_dynamic(env, S_list, phase_ids):
    T, K = env.shape
    a_star = int(np.argmin(env.sum(axis=0)))
    C_static = float(env[:, a_star].sum())
    C_oracle, prev, oracle_arms = 0.0, None, []
    for ph in np.unique(phase_ids):
        m = phase_ids == ph
        a = int(np.argmin(env[m].sum(axis=0)))
        C_oracle += float(env[m][:, a].sum())
        if prev is not None and a != prev:
            C_oracle += float(S_list[ph][prev, a])
        prev = a; oracle_arms.append(a)
    return a_star, C_static, C_oracle, oracle_arms

def run_s1_s2():
    grid = load_grid()
    lam_E, Lref, Eref = fix_lambda_E(grid)
    print(f"lambda_E = {lam_E:.3f} (med L {Lref:.2f} ms / med E {Eref:.2f} mJ), "
          f"lambda_D = {LAMBDA_D:.1f} ms")
    out = ROOT / "output" / f"replay_results_{HW_ID}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario","model","seed","arms","oracle_arms","inversion",
                    "C_static","C_oracle","selector","C_T","switches","delta_vs_static"])
        for kind, model, seed in itertools.product(
                ["S1","S2"],
                ["vit_tiny_patch16_224","vit_small_patch16_224",
                 "vit_base_patch16_224","deit_small_patch16_224"],
                [42,1234,2024]):
            try:
                env, S_list, pid, arms, phases = build_scenario(grid, seed, model, kind)
            except FileNotFoundError:
                continue
            if env.shape[1] < 2:
                continue
            K = env.shape[1]
            Sbar = np.mean(S_list, axis=0)
            scale = float(np.median(env))
            a_star, C_st, C_or, oarms = baselines_dynamic(env, S_list, pid)
            inv = len(set(oarms)) > 1
            mk = {"random": RandomSelector(K, switch_costs=Sbar, rng=np.random.default_rng(seed)),
                  "eps": EpsGreedy(K, switch_costs=Sbar, rng=np.random.default_rng(seed)),
                  "SW-UCB": SWUCB(K, tau=300, switch_costs=Sbar),
                  "D-UCB": DUCB(K, gamma=0.995, switch_costs=Sbar),
                  "EXP3.S": EXP3S(K, cost_scale=2*scale, switch_costs=Sbar,
                                  rng=np.random.default_rng(seed)),
                  "GLR-klUCB": GLRklUCB(K, switch_costs=Sbar)}
            for name, sel in mk.items():
                C_T, sw = replay_dynamic(env, S_list, pid, sel)
                w.writerow([kind, model, seed, "|".join(arms),
                            "|".join(arms[a] for a in oarms), int(inv),
                            f"{C_st:.0f}", f"{C_or:.0f}", name,
                            f"{C_T:.0f}", sw, f"{C_st - C_T:.0f}"])
            print(f"{kind} {model.split('_')[1]:5s} s{seed}: inversion={inv} "
                  f"oracle={'>'.join(arms[a] for a in oarms)}")
    print(f"saved {out}")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "run":
    run_s1_s2()
