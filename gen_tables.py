# Generates tables/tab_accuracy.tex (Table 1 of the manuscript) from the
# accuracy-equivalence CSV written by cell7_accuracy.py.
# Columns: eager top-1 and the deviation of the three ONNX Runtime arms
# (percentage points). TorchScript and both Inductor modes are checked to
# match eager to two decimals and are omitted from the table.
import csv, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "projects" / "vit_backend_adaptive"
SRC = ROOT / "output" / "accuracy_equivalence_RTX_4500_Ada_Generation.csv"
OUT = ROOT / "manuscript" / "tables"; OUT.mkdir(parents=True, exist_ok=True)

SHOWN = ["ort_cuda_fp32", "ort_cuda_fp16", "ort_trt"]
OMITTED = ["torchscript", "inductor_default", "inductor_maxautotune"]
MODEL = {"vit_tiny_patch16_224": "ViT-T", "vit_small_patch16_224": "ViT-S",
         "vit_base_patch16_224": "ViT-B", "deit_small_patch16_224": "DeiT-S"}
MODEL_ORDER = list(MODEL)
DS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "imagenette": "Imagenette"}
DS_ORDER = list(DS)

acc = defaultdict(dict)
for r in csv.DictReader(open(SRC)):
    acc[(r["model"], r["dataset"])][r["backend"]] = float(r["top1"])

def dev(x):
    s = f"{x:+.2f}"
    if s in ("+0.00", "-0.00"):
        return "0.00"
    return s.replace("-", "$-$")

for k, d in acc.items():
    for b in OMITTED:
        if f"{d[b] - d['eager']:.2f}" not in ("0.00", "-0.00"):
            sys.exit(f"{k} {b} deviates from eager; it cannot be omitted")

lines = [
    "\\begin{table}[t]",
    "\\caption{Top-1 test accuracy (\\%) of eager execution on the Ada",
    "platform and deviation of the ONNX Runtime arms (percentage points).",
    "TorchScript and both Inductor modes match eager to two decimals and are",
    "omitted.}",
    "\\label{tab:accuracy}\\centering\\footnotesize",
    "\\begin{tabular}{@{}llrrrr@{}}",
    "\\toprule",
    "Model & Data & Eager & FP32 & FP16 & TRT \\\\",
    "\\midrule",
]
for mn in MODEL_ORDER:
    for ds in DS_ORDER:
        d = acc[(mn, ds)]
        e = d["eager"]
        lines.append(f"{MODEL[mn]} & {DS[ds]} & {e:.2f} & "
                     + " & ".join(dev(d[b] - e) for b in SHOWN) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(OUT / "tab_accuracy.tex").write_text("\n".join(lines) + "\n")
mx = max(abs(acc[k][b] - acc[k]["eager"]) for k in acc for b in SHOWN + OMITTED)
print(f"written {OUT / 'tab_accuracy.tex'} | max |deviation| = {mx:.3f} pp")
