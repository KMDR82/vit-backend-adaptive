# Fig 3 — contention: median cost (over 4 models, 3 seeds) per backend vs
# duty level (x labels carry measured watts), panels (a) b1 (b) b8. Log y.
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home()/"projects"/"vit_backend_adaptive"
FIG = ROOT/"fig"
LAM_E, LAM_D = 0.041, 1000.0/60
BE = ["eager","torchscript","inductor_default","inductor_maxautotune",
      "ort_cuda_fp32","ort_cuda_fp16","ort_trt"]
BE_LBL = {"eager":"Eager","torchscript":"TorchScript","inductor_default":"Ind-def",
          "inductor_maxautotune":"Ind-MA","ort_cuda_fp32":"ORT-fp32",
          "ort_cuda_fp16":"ORT-fp16","ort_trt":"ORT-TRT"}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
STYLE = {be: dict(color=COLORS[i], marker="osD^vP*"[i], ms=3.5, lw=1.2)
         for i, be in enumerate(BE)}

cal = {r["duty"]: float(r["power_mean_W"]) for r in
       csv.DictReader(open(ROOT/"output"/"contention_calibration_RTX_4500_Ada_Generation.csv"))}
LVLS = ["0","25","50","100"]
XLBL = [f"{l}%\n({cal[l]:.0f} W)" for l in LVLS]

rows = [r for r in csv.DictReader(open(ROOT/"output"/"contention_grid_RTX_4500_Ada_Generation.csv"))
        if r["status"]=="ok"]
def med_cost(be, b, lvl):
    v = [float(r["lat_med_ms"]) + LAM_E*float(r["energy_mJ_per_inf"])
         + LAM_D*float(r["miss60"]) for r in rows
         if r["backend"]==be and r["batch"]==b and r["contention"]==lvl]
    return np.median(v)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, b, lab in zip(axes, ["1","8"], ["(a)","(b)"]):
    for be in BE:
        ys = [med_cost(be, b, l) for l in LVLS]
        ax.plot(range(4), ys, label=BE_LBL[be], **STYLE[be])
    ax.set_yscale("log")
    ax.set_xticks(range(4)); ax.set_xticklabels(XLBL, fontsize=7.5)
    ax.set_xlabel("Contention level (measured board power)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.text(0.5, -0.34, lab, transform=ax.transAxes, ha="center", fontsize=10)
axes[0].set_ylabel("Median cost (ms-equiv., log)", fontsize=9)
axes[1].legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
for ext in ["png","pdf"]:
    fig.savefig(FIG/f"fig3_contention.{ext}", dpi=300, bbox_inches="tight")
with open(FIG/"fig3_contention_data.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["batch","duty","power_W","backend","median_cost"])
    for b in ["1","8"]:
        for l in LVLS:
            for be in BE:
                w.writerow([b, l, f"{cal[l]:.1f}", be, f"{med_cost(be,b,l):.3f}"])
print("saved fig/fig3_contention.{png,pdf} + data csv")
