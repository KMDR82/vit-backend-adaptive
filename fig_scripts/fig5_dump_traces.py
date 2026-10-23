# Fig 5 input — step-by-step cumulative cost traces for T4/S1/vit_base/s42:
# oracle, GLR-UCB, D-UCB, SW-UCB vs best-static. Output:
# fig/fig5_traces_data.csv (step, phase, cum_delta per policy, switch flags)
import os, csv, sys
from pathlib import Path
import numpy as np

ROOT = Path.home()/"projects"/"vit_backend_adaptive"
os.environ["HW"] = "Tesla_T4"
sys.path.insert(0, str(ROOT))
import cell13_replay as c13
from bandit_selectors import GLRklUCB, DUCB, SWUCB

grid = c13.load_grid()
model, kind, seed = "vit_base_patch16_224", "S1", 42
env, S_list, pid, arms, phases = c13.build_scenario(grid, seed, model, kind)
T, K = env.shape
a_star = int(np.argmin(env.sum(axis=0)))
static_cum = np.cumsum(env[:, a_star])

def trace(policy):
    cum, prev, out, sw = 0.0, None, [], []
    for t in range(T):
        a = policy.select(); c = float(env[t, a]); policy.observe(a, c); cum += c
        if prev is not None and a != prev:
            cum += float(S_list[pid[t]][prev, a]); sw.append(t)
        prev = a; out.append(cum)
    return np.array(out), sw

def oracle_trace():
    cum, prev, out = 0.0, None, []
    for ph in np.unique(pid):
        m = pid == ph
        a = int(np.argmin(env[m].sum(axis=0)))
        if prev is not None and a != prev:
            cum += float(S_list[ph][prev, a])
        for c in env[m][:, a]:
            cum += float(c); out.append(cum)
        prev = a
    return np.array(out)

pols = {"GLR-UCB": GLRklUCB(K, switch_costs=np.mean(S_list,0)),
        "D-UCB": DUCB(K, gamma=0.995, switch_costs=np.mean(S_list,0)),
        "SW-UCB": SWUCB(K, tau=300, switch_costs=np.mean(S_list,0))}
results = {"oracle": (oracle_trace(), [])}
for n, p in pols.items(): results[n] = trace(p)

FIG = ROOT/"fig"
with open(FIG/"fig5_traces_data.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["step","phase"] + [f"delta_{n}" for n in results] +
               [f"switch_{n}" for n in results])
    for t in range(T):
        row = [t, int(pid[t])]
        for n in results: row.append(f"{results[n][0][t]-static_cum[t]:.1f}")
        for n in results: row.append(1 if t in results[n][1] else 0)
        w.writerow(row)
print(f"arms: {arms} | a*={arms[a_star]} | phases: {phases}")
for n in results:
    print(f"{n:8s} final delta {static_cum[-1]-results[n][0][-1]:10.0f} "
          f"switches {len(results[n][1])}")
print("saved fig/fig5_traces_data.csv")
