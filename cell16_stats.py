# Cell 16 — statistics. Module 1: break-even phase-length analysis.
# T* = S_roundtrip / margin_per_step, for every bucket whose local winner
# differs from the platform's global static arm. Cold S = max setup_s over
# seeds, warm S = min. Outputs: output/breakeven_{HW}.csv + printed table.
import csv, sys, os
from pathlib import Path
import numpy as np
from collections import defaultdict

ROOT = Path.home()/"projects"/"vit_backend_adaptive"
LAM = {"RTX_4500_Ada_Generation": 0.041, "Tesla_T4": 0.120}   # lambda_E per platform (Sect. 5.3)
LAM_D = 1000.0/60

def analyze(hw):
    rows = [r for r in csv.DictReader(open(ROOT/"output"/f"static_grid_{hw}.csv"))
            if r["status"]=="ok"]
    lam_E = LAM[hw]
    cost = defaultdict(lambda: defaultdict(list))
    setup = defaultdict(list)
    for r in rows:
        k = (r["model"], int(r["batch"]), int(r["res"]))
        c = float(r["lat_med_ms"]) + lam_E*float(r["energy_mJ_per_inf"]) + LAM_D*float(r["miss60"])
        cost[k][r["backend"]].append(c)
        setup[(k, r["backend"])].append(float(r["setup_s"]))
    mean_cost = {k: {be: np.mean(v) for be, v in d.items()} for k, d in cost.items()}
    # global static arm = min total cost across all buckets (equal weights)
    tot = defaultdict(float)
    for k, d in mean_cost.items():
        for be, c in d.items(): tot[be] += c
    a_star = min(tot, key=tot.get)
    out = []
    for k, d in sorted(mean_cost.items()):
        s = sorted(d.items(), key=lambda x: x[1])
        local, c_local = s[0]
        if local == a_star: continue
        m = d[a_star] - c_local                    # margin per step (ms-equivalent)
        S_cold = (max(setup[(k, local)]) + max(setup[(k, a_star)])) * 1000
        S_warm = (min(setup[(k, local)]) + min(setup[(k, a_star)])) * 1000
        out.append(dict(bucket=f"{k[0].split('_')[1]}/b{k[1]}/r{k[2]}",
            local_winner=local, global_static=a_star,
            margin_ms=round(m,3), margin_pct=round(100*m/c_local,1),
            S_cold_ms=round(S_cold), S_warm_ms=round(S_warm),
            T_cold=round(S_cold/m) if m>0 else None,
            T_warm=round(S_warm/m) if m>0 else None))
    print(f"\n=== {hw} | global static arm: {a_star} ===")
    if not out:
        print("  hicbir kovada yerel kazanan global koldan farkli degil -> T* = inf (tersinme yok)")
    for o in out:
        print(f"  {o['bucket']}: {o['local_winner']} yener (marj {o['margin_ms']} ms/adim, %{o['margin_pct']})")
        print(f"    S gidis-donus: soguk {o['S_cold_ms']} ms / sicak {o['S_warm_ms']} ms")
        print(f"    T* basabas faz: soguk {o['T_cold']} adim / sicak {o['T_warm']} adim  (senaryo fazi: 300)")
    with open(ROOT/"output"/f"breakeven_{hw}.csv","w",newline="") as f:
        if out:
            w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
        else:
            f.write("no_bucket_deviates_from_global_static\n")
    return out

if __name__ == "__main__":
    for hw in ["RTX_4500_Ada_Generation", "Tesla_T4"]:
        analyze(hw)

# ================= Module 2: selector significance tests =================
# Per platform: selector x 24-run matrix of delta_vs_static ->
# Friedman test (do selectors differ?), Nemenyi CD (which pairs differ),
# Wilcoxon signed-rank per selector vs 0 (does it beat best-static?).
from scipy import stats

SELECTORS = ["random","eps","SW-UCB","D-UCB","EXP3.S","GLR-klUCB"]
PAPER_NAME = {"GLR-klUCB": "GLR-UCB"}   # rename at presentation (honesty pass)

def module2(hw):
    path = ROOT/"output"/f"replay_results_{hw}.csv"
    rows = list(csv.DictReader(open(path)))
    runs = sorted({(r["scenario"], r["model"], r["seed"]) for r in rows})
    M = np.full((len(runs), len(SELECTORS)), np.nan)
    SW = np.full_like(M, np.nan)
    for r in rows:
        i = runs.index((r["scenario"], r["model"], r["seed"]))
        j = SELECTORS.index(r["selector"])
        M[i, j] = float(r["delta_vs_static"]); SW[i, j] = float(r["switches"])
    assert not np.isnan(M).any()
    print(f"\n=== {hw}: selector table (24 runs) ===")
    print(f"{'selector':10s} {'delta med':>12s} {'delta mean':>12s} {'sw med':>7s} "
          f"{'Wilcoxon p':>11s}")
    wilc = {}
    for j, sel in enumerate(SELECTORS):
        stat, p = stats.wilcoxon(M[:, j])          # H0: median delta = 0
        wilc[sel] = p
        nm = PAPER_NAME.get(sel, sel)
        print(f"{nm:10s} {np.median(M[:,j]):12.0f} {M[:,j].mean():12.0f} "
              f"{np.median(SW[:,j]):7.0f} {p:11.2e}")
    fr_stat, fr_p = stats.friedmanchisquare(*[M[:, j] for j in range(len(SELECTORS))])
    print(f"Friedman: chi2={fr_stat:.1f}, p={fr_p:.2e}")
    # Nemenyi CD via studentized range (alpha=0.05)
    n, k = M.shape
    ranks = np.array([stats.rankdata(-row) for row in M])   # higher delta = better rank 1
    avg_rank = ranks.mean(axis=0)
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    order = np.argsort(avg_rank)
    print(f"Nemenyi CD={cd:.2f} | avg ranks (best first):")
    for j in order:
        nm = PAPER_NAME.get(SELECTORS[j], SELECTORS[j])
        print(f"  {nm:10s} {avg_rank[j]:.2f}")
    with open(ROOT/"output"/f"selector_stats_{hw}.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["selector","delta_median","delta_mean","switches_median",
                    "wilcoxon_p","avg_rank","friedman_chi2","friedman_p","nemenyi_cd"])
        for j, sel in enumerate(SELECTORS):
            w.writerow([PAPER_NAME.get(sel,sel), f"{np.median(M[:,j]):.1f}",
                        f"{M[:,j].mean():.1f}", f"{np.median(SW[:,j]):.0f}",
                        f"{wilc[sel]:.3e}", f"{avg_rank[j]:.2f}",
                        f"{fr_stat:.2f}", f"{fr_p:.3e}", f"{cd:.3f}"])
if __name__ == "__main__":
    for hw in ["RTX_4500_Ada_Generation", "Tesla_T4"]:
        module2(hw)

# ============ Module 3: phase-length sweep (positive control) ============
# Empirical test of the break-even prediction (T*_warm ~ 789 on T4/base):
# replay T4/S1/vit_base with PHASE_LEN in {300,600,900,1500,3000} under
# cold-S and warm-S regimes; track oracle / GLR / D-UCB delta vs static.
# Bootstrap note: 3000 steps resample the 200 measured samples ~15x (stated
# in the paper). Output: output/phase_sweep_Tesla_T4.csv
sys.path.insert(0, str(ROOT))
from bandit_selectors import DUCB, GLRklUCB
import cell13_replay as c13

def module3():
    hw = "Tesla_T4"
    os.environ["HW"] = hw
    import importlib; importlib.reload(c13)
    grid = c13.load_grid()
    model, kind = "vit_base_patch16_224", "S1"
    # cold/warm S matrices per bucket: max/min setup_s over seeds
    rows = [r for r in csv.DictReader(open(ROOT/"output"/f"static_grid_{hw}.csv"))
            if r["status"]=="ok"]
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], int(r["batch"]), int(r["res"]), r["backend"])].append(float(r["setup_s"]))
    def S_of(regime):
        f = max if regime=="cold" else min
        S = {}
        for (m,b,res) in {(model,b,224) for b in [1,8,32]}:
            mat = np.zeros((len(c13.BACKENDS),)*2)
            for j,t in enumerate(c13.BACKENDS):
                v = agg.get((m,b,res,t))
                if v: mat[:,j] = f(v)*1000.0
            np.fill_diagonal(mat,0.0); S[(m,b,res)] = mat
        return S
    out = []
    print(f"\n=== phase-length sweep: {hw} / S1 / vit_base ===")
    print(f"{'regime':6s} {'L':>5s} {'oracle_d':>10s} {'GLR_d':>10s} {'DUCB_d':>10s}")
    for regime in ["cold","warm"]:
        Sdict = S_of(regime)
        for L in [300, 600, 900, 1500, 3000]:
            c13.PHASE_LEN = L
            # monkey-patch load_switch_costs to the regime matrices
            c13.load_switch_costs = lambda Sd=Sdict: Sd
            deltas = defaultdict(list)
            for seed in [42, 1234, 2024]:
                env, S_list, pid, arms, _ = c13.build_scenario(grid, seed, model, kind)
                a_star, C_st, C_or, _ = c13.baselines_dynamic(env, S_list, pid)
                deltas["oracle"].append(C_st - C_or)
                K = env.shape[1]; Sbar = np.mean(S_list, axis=0)
                for name, sel in [("GLR", GLRklUCB(K, switch_costs=Sbar)),
                                  ("DUCB", DUCB(K, gamma=0.995, switch_costs=Sbar))]:
                    C_T, _ = c13.replay_dynamic(env, S_list, pid, sel)
                    deltas[name].append(C_st - C_T)
            o, g, d = (np.mean(deltas[k]) for k in ["oracle","GLR","DUCB"])
            print(f"{regime:6s} {L:5d} {o:10.0f} {g:10.0f} {d:10.0f}")
            out.append(dict(regime=regime, phase_len=L, oracle_delta=round(o),
                            glr_delta=round(g), ducb_delta=round(d)))
    with open(ROOT/"output"/"phase_sweep_Tesla_T4.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print("saved output/phase_sweep_Tesla_T4.csv")

if __name__ == "__main__":
    module3()

# ================= Module 4: lambda sensitivity =================
# Does the conclusion depend on the weights? Sweep lambda_E x {0,.5,1,2,4}
# (of the platform's lambda_E) and lambda_D in {0, 1000/60, 1000/30};
# count bucket winners per combo. Output: output/lambda_sensitivity.csv
def module4():
    out = []
    for hw in ["RTX_4500_Ada_Generation", "Tesla_T4"]:
        rows = [r for r in csv.DictReader(open(ROOT/"output"/f"static_grid_{hw}.csv"))
                if r["status"]=="ok"]
        base_lam = LAM[hw]
        print(f"\n=== {hw}: lambda sensitivity (bucket winners / 24) ===")
        print(f"{'lamE_mult':>9s} {'lamD':>6s}  winners")
        for mE in [0, 0.5, 1, 2, 4]:
            for lamD in [0.0, 1000/60, 1000/30]:
                cost = defaultdict(lambda: defaultdict(list))
                for r in rows:
                    c = (float(r["lat_med_ms"]) + mE*base_lam*float(r["energy_mJ_per_inf"])
                         + lamD*float(r["miss60"]))
                    cost[(r["model"], r["batch"], r["res"])][r["backend"]].append(c)
                win = defaultdict(int); flips = []
                for k, d in cost.items():
                    m = {be: np.mean(v) for be, v in d.items()}
                    w = min(m, key=m.get); win[w] += 1
                    if w != "ort_trt": flips.append((k, w))
                wstr = ", ".join(f"{k}:{v}" for k, v in sorted(win.items(), key=lambda x:-x[1]))
                fstr = "; ".join(f"{k[0].split('_')[1]}/b{k[1]}/r{k[2]}->{w}" for k, w in flips)
                print(f"{mE:9.1f} {lamD:6.1f}  {wstr}" + (f"  [{fstr}]" if flips else ""))
                out.append(dict(hw=hw, lamE_mult=mE, lamD=round(lamD,1),
                                trt_wins=win.get("ort_trt",0),
                                flips=fstr if flips else ""))
    with open(ROOT/"output"/"lambda_sensitivity.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print("saved output/lambda_sensitivity.csv")

if __name__ == "__main__":
    module4()
