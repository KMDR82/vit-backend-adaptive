# Fig 4 — phase-length sweep (T4/S1/vit_base): reads output/phase_sweep_Tesla_T4.csv
import csv, shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home()/"projects"/"vit_backend_adaptive"
FIG = ROOT/"fig"
rows = list(csv.DictReader(open(ROOT/"output"/"phase_sweep_Tesla_T4.csv")))
L = sorted({int(r["phase_len"]) for r in rows})
def series(reg, col):
    return [float(next(r[col] for r in rows
            if r["regime"]==reg and int(r["phase_len"])==l))/1000.0 for l in L]
T_STAR = 789
styles = {"oracle": dict(color="#000000", marker="o", ls="-"),
          "GLR-UCB": dict(color="#1f77b4", marker="s", ls="--"),
          "D-UCB": dict(color="#d62728", marker="^", ls="-.")}
cols = {"oracle": "oracle_delta", "GLR-UCB": "glr_delta", "D-UCB": "ducb_delta"}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, regime, lab in zip(axes, ["cold","warm"], ["(a)","(b)"]):
    for name, col in cols.items():
        ax.plot(L, series(regime, col), label=name, lw=1.4, ms=4, **styles[name])
    ax.axhline(0, color="gray", lw=0.8)
    if regime == "warm":
        ax.axvline(T_STAR, color="gray", lw=0.9, ls=":")
        ax.annotate(r"$T^{*}_{\mathrm{warm}}=789$", xy=(T_STAR, 0), xytext=(950, 12),
                    fontsize=8, color="dimgray",
                    arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.7))
    ax.set_xlabel("Phase length (steps)", fontsize=9)
    ax.set_xticks(L); ax.tick_params(labelsize=8)
    ax.text(0.5, -0.30, lab, transform=ax.transAxes, ha="center", fontsize=10)
axes[0].set_ylabel(r"$\Delta_T$ vs. best-static ($\times 10^3$)", fontsize=9)
axes[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()
for ext in ["png","pdf"]:
    fig.savefig(FIG/f"fig4_phase_sweep.{ext}", dpi=300, bbox_inches="tight")
shutil.copy(ROOT/"output"/"phase_sweep_Tesla_T4.csv", FIG/"fig4_phase_sweep_data.csv")
print("saved fig/fig4_phase_sweep.{png,pdf} + data csv")
