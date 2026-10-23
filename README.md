# vit-backend-adaptive

Code and measurement data for the manuscript

> A. Akkaya, "When does adaptive backend selection beat static profiling? A cross-hardware study of real-time Vision Transformer inference under non-stationary load", submitted to the *Journal of Real-Time Image Processing*.

The study measures seven execution backends (PyTorch eager, TorchScript, `torch.compile`/Inductor in two modes, ONNX Runtime CUDA FP32/FP16, ONNX Runtime TensorRT) for four ViT models on an RTX 4500 Ada and a Tesla T4. It then replays the measured traces under non-stationary scenarios to test four switching-aware non-stationary bandit selectors and two baselines against the best static backend.

## Repository layout

| Path | Content |
| --- | --- |
| `vit_backend_adaptive.ipynb` | Environment probe and NVML power-counter characterization |
| `cell7_accuracy.py` | Output equivalence: top-1 accuracy of every backend |
| `fix_fp16.py` | First FP16 regeneration (ViT-Tiny) with the ONNX Runtime transformer optimizer; the study's FP16 graphs are built the same way inside `cell7_accuracy.py` and `cell8_grid.py` |
| `cell8_grid.py` | Static grid: backends x models x batch {1, 8, 32} x resolution {224, 384} x 3 seeds (CIFAR-10 checkpoints) |
| `cell11_contention.py`, `contender.py` | Contention grid with a co-located synthetic GPU load at 0/25/50/100% duty (Ada) |
| `rerun224.py` | Re-measurement of 168 Ada configurations (224 px, batch 1 and 8) |
| `bandit_selectors.py` | SW-UCB, D-UCB, EXP3.S, GLR-UCB, epsilon-greedy and random, all switching-aware |
| `cell13_replay.py` | Replay of scenarios S1 (batch) and S2 (resolution) on both GPUs |
| `cell14_s3.py` | Replay of scenario S3 (idle/saturated contention, Ada) |
| `cell16_stats.py` | Wilcoxon, Friedman and Nemenyi statistics |
| `cell17_resident_tail.py` | Replay with all engines resident (switching cost 0); p99 latency and deadline-miss rates per bucket |
| `gen_tables.py` | Manuscript tables |
| `fig_scripts/` | Manuscript figures |
| `fig/*.csv` | Data behind each figure |
| `output/*.csv`, `output/noise_floor_*.json` | Aggregated results: static grid, contention grid and calibration, re-measurement, accuracy, setup costs, replay results, statistics, tail latency; NVML counter characterization |
| `output/traces.tar.gz` | Per-configuration latency traces (`.npz`, 200 timed samples each) for both GPUs and the saturated-contention level |

## Software

Ada platform: Python 3.11, PyTorch 2.5.1 (CUDA 12.1, cuDNN 9.1), TorchVision 0.20.1.
T4 platform (Kaggle): Python 3.12, PyTorch 2.10 (CUDA 12.8).
Both: timm 1.0.29, ONNX Runtime GPU 1.22.0, TensorRT 10.9, nvidia-ml-py, numpy, scipy, pandas, matplotlib.

## Reproducing the analysis

The scripts expect the repository at `~/projects/vit_backend_adaptive` (edit `ROOT` at the top of a script to change this). The replay and statistics run on the released data and need no GPU:

```bash
git clone https://github.com/KMDR82/vit-backend-adaptive ~/projects/vit_backend_adaptive
cd ~/projects/vit_backend_adaptive
tar -xzf output/traces.tar.gz -C output/
HW=RTX_4500_Ada_Generation python cell13_replay.py run   # S1, S2 (Ada)
HW=Tesla_T4 python cell13_replay.py run                  # S1, S2 (T4)
python cell14_s3.py                                      # S3 (Ada)
python cell16_stats.py                                   # break-even, tests, sensitivity
HW=RTX_4500_Ada_Generation python cell17_resident_tail.py
HW=Tesla_T4 python cell17_resident_tail.py
python gen_tables.py                                     # Table 1
```

Re-running the measurements (`cell7`, `cell8`, `cell11`, `rerun224`) requires the fine-tuned models, the datasets (CIFAR-10, CIFAR-100, Imagenette) and the corresponding GPU. Model weights, ONNX graphs, TensorRT engines and datasets are not included because of their size; the scripts regenerate them.

## Contact

Ahmet Akkaya, Bandırma Onyedi Eylül University, aakkaya@bandirma.edu.tr, ORCID [0000-0003-4836-2310](https://orcid.org/0000-0003-4836-2310)
