# Fig 2 — static landscape: cost ratio to bucket winner, 7 backends x 24 buckets,
# panels (a) Ada (b) T4. Winner cells marked with a dot.
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
BE = ["eager","torchscript","inductor_default","inductor_maxautotune",
      "ort_cuda_fp32","ort_cuda_fp16","ort_trt"]
BE_LBL = ["Eager","TorchScript","Ind-def","Ind-MA","ORT-fp32","ORT-fp16","ORT-TRT"]
MODELS = ["vit_tiny_patch16_224","vit_small_patch16_224",
          "deit_small_patch16_224","vit_base_patch16_224"]
M_LBL = {"vit_tiny_patch16_224":"T","vit_small_patch16_224":"S",
         "deit_small_patch16_224":"D","vit_base_patch16_224":"B"}

def matrix(hw):
    rows = [r for r in csv.DictReader(open(ROOT/"output"/f"static_grid_{hw}.csv"))
            if r["status"]=="ok"]
    cost = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = (float(r["lat_med_ms"]) + LAM[hw]*float(r["energy_mJ_per_inf"])
             + LAM_D*float(r["miss60"]))
        cost[(r["model"], int(r["batch"]), int(r["res"]))][r["backend"]].append(c)
    buckets = [(m,b,res) for m in MODELS for res in (224,384) for b in (1,8,32)]
    M = np.zeros((len(BE), len(buckets)))
    for j,k in enumerate(buckets):
        means = {be: np.mean(cost[k][be]) for be in BE}
        lo = min(means.values())
        for i,be in enumerate(BE): M[i,j] = means[be]/lo
    labels = [f"{M_LBL[m]}{b}\n{res}" for (m,b,res) in buckets]
    return M, labels

fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2))
fig.subplots_adjust(hspace=0.85)
for ax, hw, lab in zip(axes, ["RTX_4500_Ada_Generation","Tesla_T4"], ["(a)","(b)"]):
    M, labels = matrix(hw)
    im = ax.imshow(np.log2(M), aspect="auto", cmap="viridis_r",
                   vmin=0, vmax=np.log2(8))
    win = M.argmin(axis=0)
    ax.scatter(range(M.shape[1]), win, s=12, c="white", edgecolors="black",
               linewidths=0.4, zorder=3)
    ax.set_yticks(range(len(BE))); ax.set_yticklabels(BE_LBL, fontsize=7)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(labels, fontsize=5.5)
    ax.text(0.5, -0.52, lab, transform=ax.transAxes, ha="center", fontsize=10)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                  ticks=[0,1,2,3], label=r"$\log_2$(cost / bucket best)")
cb.ax.tick_params(labelsize=7); cb.set_label(r"$\log_2$(cost / bucket best)", fontsize=8)
for ext in ["png","pdf"]:
    fig.savefig(FIG/f"fig2_static_landscape.{ext}", dpi=300, bbox_inches="tight")
# per-figure data
with open(FIG/"fig2_static_landscape_data.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["hw","backend","bucket","cost_ratio"])
    for hw in ["RTX_4500_Ada_Generation","Tesla_T4"]:
        M, labels = matrix(hw)
        for i,be in enumerate(BE):
            for j,lb in enumerate(labels):
                w.writerow([hw, be, lb.replace("\n","/"), f"{M[i,j]:.4f}"])
print("saved fig/fig2_static_landscape.{png,pdf} + data csv")
