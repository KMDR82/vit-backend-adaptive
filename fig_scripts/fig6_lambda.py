# Fig 6 (v2) — robustness of TRT dominance to objective weighting.
# Per bucket: TRT margin over best non-TRT arm (%) vs lambda_E multiplier.
# 24 thin gray trajectories + median (thick); flip trajectories colored.
import csv, shutil
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home()/"projects"/"vit_backend_adaptive"
FIG = ROOT/"fig"
LAM = {"RTX_4500_Ada_Generation": 0.041, "Tesla_T4": 0.120}
LAM_D = 1000.0/60
MULTS = [0, 0.5, 1, 2, 4]
SHORT = {"vit_tiny_patch16_224":"T","vit_small_patch16_224":"S",
         "deit_small_patch16_224":"D","vit_base_patch16_224":"B"}

def margins(hw):
    rows = [r for r in csv.DictReader(open(ROOT/"output"/f"static_grid_{hw}.csv"))
            if r["status"]=="ok"]
    raw = defaultdict(lambda: defaultdict(list))
    for r in rows:
        raw[(r["model"], int(r["batch"]), int(r["res"]))][r["backend"]].append(
            (float(r["lat_med_ms"]), float(r["energy_mJ_per_inf"]), float(r["miss60"])))
    buckets = sorted(raw)
    out = {}
    for k in buckets:
        traj = []
        for m in MULTS:
            c = {be: np.mean([L + m*LAM[hw]*E + LAM_D*D for (L,E,D) in v])
                 for be, v in raw[k].items()}
            trt = c.pop("ort_trt")
            best_other = min(c.values())
            traj.append(100*(best_other - trt)/trt)   # >0: TRT wins
        out[k] = traj
    return out

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for ax, (hw, short), lab in zip(axes,
        [("RTX_4500_Ada_Generation","Ada"), ("Tesla_T4","T4")], ["(a)","(b)"]):
    tr = margins(hw)
    med = np.median(np.array(list(tr.values())), axis=0)
    for k, y in tr.items():
        if min(y) < 0:
            name = f"{SHORT[k[0]]}{k[1]}/{k[2]}"
            ax.plot(MULTS, y, color="#d62728", lw=1.4, marker="o", ms=3, zorder=3)
            yi = min(y.index(min(y)), 2)   # keep label inside axes
            ax.annotate(name, (MULTS[yi], y[yi]), textcoords="offset points",
                        xytext=(6, -11), fontsize=7, color="#d62728")
        else:
            ax.plot(MULTS, y, color="gray", lw=0.55, alpha=0.45, zorder=1)
    ax.plot(MULTS, med, color="black", lw=1.8, zorder=2)
    ax.annotate("median", (MULTS[-1], med[-1]), textcoords="offset points",
                xytext=(-30, 6), fontsize=7)
    ax.axhline(0, color="#d62728", lw=0.8, ls="--")
    ax.set_xticks(MULTS); ax.set_xticklabels([f"{m}$\\times$" for m in MULTS])
    ax.set_xlabel(r"$\lambda_E$ multiplier", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.5, -0.30, lab, transform=ax.transAxes, ha="center", fontsize=10)
axes[0].set_ylabel("TRT margin over best\nnon-TRT arm (%)", fontsize=8.5)
fig.tight_layout()
for ext in ["png","pdf"]:
    fig.savefig(FIG/f"fig6_lambda.{ext}", dpi=300, bbox_inches="tight")
# per-figure data
with open(FIG/"fig6_lambda_data.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["hw","bucket"]+[f"margin_pct_{m}x" for m in MULTS])
    for hw,_ in [("RTX_4500_Ada_Generation","Ada"),("Tesla_T4","T4")]:
        for k, y in margins(hw).items():
            w.writerow([hw, f"{SHORT[k[0]]}{k[1]}/{k[2]}"]+[f"{v:.2f}" for v in y])
print("saved fig/fig6_lambda.{png,pdf} + data csv")
