# Fig 5 — (a) cumulative delta vs best-static over time (T4/S1/base/s42),
# phase boundaries dotted; (b) Nemenyi CD diagram, both platforms.
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home()/"projects"/"vit_backend_adaptive"
FIG = ROOT/"fig"
rows = list(csv.DictReader(open(FIG/"fig5_traces_data.csv")))
t = np.array([int(r["step"]) for r in rows])
POLS = ["oracle","GLR-UCB","D-UCB","SW-UCB"]
STY = {"oracle": dict(color="k", ls="-", lw=1.5),
       "GLR-UCB": dict(color="#1f77b4", ls="--", lw=1.3),
       "D-UCB": dict(color="#d62728", ls="-.", lw=1.3),
       "SW-UCB": dict(color="#2ca02c", ls=":", lw=1.5)}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                         gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axes[0]
for n in POLS:
    d = np.array([float(r[f"delta_{n}"]) for r in rows]) / 1000.0
    ax.plot(t, d, label=n, **STY[n])    # positive = extra cost vs static
for pb in [300, 600, 900, 1200]:
    ax.axvline(pb, color="gray", lw=0.7, ls=":")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Step", fontsize=9); ax.set_ylabel(r"$C_T(\pi)-C_T(a^\star)$ ($\times 10^3$)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.tick_params(labelsize=8)
ax.text(0.5, -0.30, "(a)", transform=ax.transAxes, ha="center", fontsize=10)

# (b) Nemenyi CD diagram from selector_stats CSVs
ax = axes[1]
HWL = [("RTX_4500_Ada_Generation","Ada"), ("Tesla_T4","T4")]
yy = {"Ada": 1.0, "T4": 0.0}
cd = None
for hw, short in HWL:
    st = list(csv.DictReader(open(ROOT/"output"/f"selector_stats_{hw}.csv")))
    cd = float(st[0]["nemenyi_cd"])
    for i, r in enumerate(sorted(st, key=lambda q: float(q["avg_rank"]))):
        x = float(r["avg_rank"]); y = yy[short]
        ax.plot(x, y, "o", ms=4, color="#333333")
        ax.annotate(r["selector"], (x, y), textcoords="offset points",
                    xytext=(0, 8 if i % 2 == 0 else -15),
                    ha="center", fontsize=6.5)
    ax.text(6.35, yy[short], short, fontsize=8, va="center")
ax.plot([1, 1 + cd], [1.55, 1.55], lw=2.2, color="#1f77b4")
ax.text(1 + cd/2, 1.66, f"CD = {cd:.2f}", ha="center", fontsize=7, color="#1f77b4")
ax.set_xlim(0.6, 6.9); ax.set_ylim(-0.55, 1.95)
ax.set_xlabel("Average rank (lower is better)", fontsize=9)
ax.set_yticks([]); ax.tick_params(labelsize=8)
for s in ["left","right","top"]: ax.spines[s].set_visible(False)
ax.text(0.5, -0.30, "(b)", transform=ax.transAxes, ha="center", fontsize=10)
fig.tight_layout()
for ext in ["png","pdf"]:
    fig.savefig(FIG/f"fig5_selectors.{ext}", dpi=300, bbox_inches="tight")
print("saved fig/fig5_selectors.{png,pdf}")
