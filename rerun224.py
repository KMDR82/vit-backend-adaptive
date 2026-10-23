# rerun224.py — re-measure the 168 static Ada configurations (r224, b1/b8,
# 7 backends x 4 models x 3 seeds) whose raw traces were overwritten by the
# contention run. Same protocol as the Ada static grid (Cell 8) / T4 grid:
# 50 warm + 200 timed iterations, >=10 s NVML energy window (10 ms sampling),
# SM clocks logged, per-seed shuffled order.
# Writes:  output/static_grid_rerun224_RTX_4500_Ada_Generation.csv  (new CSV)
#          output/static_grid_raw_RTX_4500_Ada_Generation/<key>.npz  (refills the
#          168 missing traces; the original grid CSV is never touched)
# Resume-safe: rows already in the rerun CSV are skipped.
import os, sys, site, ctypes, glob, csv, time, random, threading
from pathlib import Path

ROOT = Path.home() / "projects" / "vit_backend_adaptive"
OUT = ROOT / "output"
HW = "RTX_4500_Ada_Generation"
RAW = OUT / f"static_grid_raw_{HW}"
C100 = OUT / f"contention_raw_{HW}" / "c100"
CSV_OUT = OUT / f"static_grid_rerun224_{HW}.csv"
MDIR = ROOT / "model"
TRT_CACHE = MDIR / "trt_cache"
SEEDS, BATCHES, RES = [42, 1234, 2024], [1, 8], [224]
N_WARM, N_TIMED, ENERGY_MIN_S = 50, 200, 10.0
DS, NCLS = "cifar10", 10

# ---------- safety guard: never overwrite the c100 traces ----------
n_c100 = len(list(C100.glob("*.npz"))) if C100.exists() else 0
n_static = len(list(RAW.glob("*.npz")))
if n_c100 != 168 or n_static not in range(336, 505):
    sys.exit(f"GUARD: c100 dir has {n_c100} files (need 168), static raw has {n_static}. "
             "Run the move step first. Nothing measured.")

# ---------- library preload (as in Cell 1; nvblas excluded) ----------
sp = site.getsitepackages()[0]
for d in sorted(glob.glob(f"{sp}/nvidia/*/lib")) + [f"{sp}/tensorrt_libs"]:
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if ".so" in f and "nvblas" not in f:
                try: ctypes.CDLL(os.path.join(d, f), mode=ctypes.RTLD_GLOBAL)
                except OSError: pass

import numpy as np
import torch, timm, pynvml
import onnxruntime as ort
from timm.models.vision_transformer import checkpoint_filter_fn

pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
MODELS = {"vit_tiny_patch16_224": (3, 192), "vit_small_patch16_224": (6, 384),
          "vit_base_patch16_224": (12, 768), "deit_small_patch16_224": (6, 384)}
BACKENDS = ["eager", "torchscript", "inductor_default", "inductor_maxautotune",
            "ort_cuda_fp32", "ort_cuda_fp16", "ort_trt"]

def load_ft(mn, res=224):
    m = timm.create_model(mn, pretrained=False, num_classes=NCLS, img_size=res)
    sd = torch.load(MDIR / f"{mn}_{DS}_best.pth", map_location="cpu", weights_only=True)
    if res != 224: sd = checkpoint_filter_fn(sd, m)
    m.load_state_dict(sd); m.eval(); return m

def onnx_path(mn, res, fp16=False):
    p = MDIR / f"{mn}_{DS}_{res}.onnx"
    q = MDIR / f"{mn}_{DS}_{res}_fp16.onnx"
    target = q if fp16 else p
    if not target.exists():
        sys.exit(f"missing {target.name}: the grid's ONNX files should already exist in model/")
    return target

def make_infer(be, mn, res):
    if be in ("eager", "torchscript", "inductor_default", "inductor_maxautotune"):
        m = load_ft(mn, res).cuda()
        if be == "torchscript": m = torch.jit.script(m)
        elif be.startswith("inductor"):
            m = torch.compile(m, mode="max-autotune" if be.endswith("maxautotune") else "default",
                              dynamic=False)
        @torch.no_grad()
        def infer(x): return m(x)
        return infer, "torch"
    p = onnx_path(mn, res, fp16=(be == "ort_cuda_fp16"))
    if be == "ort_trt":
        prov = [("TensorrtExecutionProvider", {"trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(TRT_CACHE), "trt_fp16_enable": True,
                "trt_timing_cache_enable": True}), "CUDAExecutionProvider", "CPUExecutionProvider"]
        want = "TensorrtExecutionProvider"
    else:
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]; want = "CUDAExecutionProvider"
    s = ort.InferenceSession(str(p), ort.SessionOptions(), providers=prov)
    if s.get_providers()[0] != want:
        raise RuntimeError(f"{be}: EP fallback to {s.get_providers()[0]}")
    def infer(x):
        return torch.from_numpy(s.run(["logits"], {"input": x.cpu().numpy()})[0]).cuda()
    return infer, want

def gpu_mem(): return pynvml.nvmlDeviceGetMemoryInfo(H).used / 1024**2
def sm_clock(): return pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)

def one_run(seed, be, mn, b, res):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.empty_cache(); time.sleep(1.0)
    idle_mem = gpu_mem()
    t0s = time.perf_counter()
    infer, ep = make_infer(be, mn, res)
    x = torch.randn(b, 3, res, res, device="cuda")
    with torch.no_grad(): _ = infer(x)
    torch.cuda.synchronize()
    setup_s = time.perf_counter() - t0s
    with torch.no_grad():
        for _ in range(N_WARM): infer(x)
        torch.cuda.synchronize()
        lats, clocks = [], []
        for i in range(N_TIMED):
            t0 = time.perf_counter(); infer(x); torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
            if i % 20 == 0: clocks.append(sm_clock())
    lats = np.array(lats); mem = gpu_mem()
    samples, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(H) / 1000.0))
            time.sleep(0.01)
    th = threading.Thread(target=sampler); th.start()
    n_inf, t0 = 0, time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - t0 < ENERGY_MIN_S:
            infer(x); n_inf += 1
        torch.cuda.synchronize()
    t1 = time.perf_counter(); stop.set(); th.join()
    ts = np.array([s[0] for s in samples]); ps = np.array([s[1] for s in samples])
    m = (ts >= t0) & (ts <= t1)
    trap = getattr(np, "trapezoid", None) or np.trapz
    E = trap(ps[m], ts[m]) if m.sum() > 1 else float("nan")
    np.savez(RAW / f"{seed}_{be}_{mn}_{b}_{res}.npz", lats=lats,
             power_t=ts[m], power_w=ps[m], clocks=np.array(clocks))
    miss = {f: float((lats > 1000.0 / f).mean()) for f in (30, 60, 120)}
    return dict(status="ok", ep=ep, lat_med_ms=np.median(lats),
        lat_p5_ms=np.percentile(lats, 5), lat_p95_ms=np.percentile(lats, 95),
        lat_mean_ms=lats.mean(), energy_mJ_per_inf=E * 1000 / max(n_inf, 1) / b,
        power_mean_W=ps[m].mean() if m.sum() else float("nan"),
        n_power_updates=int((np.diff(ps[m]) != 0).sum()) if m.sum() > 1 else 0,
        energy_win_s=t1 - t0, mem_MB=mem - idle_mem, idle_mem_MB=idle_mem,
        sm_clock_mean=float(np.mean(clocks)), sm_clock_min=float(np.min(clocks)),
        miss30=miss[30], miss60=miss[60], miss120=miss[120], setup_s=setup_s)

COLS = ["lat_med_ms","lat_p5_ms","lat_p95_ms","lat_mean_ms","energy_mJ_per_inf",
        "power_mean_W","n_power_updates","energy_win_s","mem_MB","idle_mem_MB",
        "sm_clock_mean","sm_clock_min","miss30","miss60","miss120","setup_s"]
if not CSV_OUT.exists():
    CSV_OUT.write_text("seed,backend,model,batch,res,status,ep," + ",".join(COLS) + "\n")
done = {(r["seed"], r["backend"], r["model"], r["batch"], r["res"])
        for r in csv.DictReader(open(CSV_OUT)) if r["status"] == "ok"}
print(f"rerun224: {len(done)}/168 already done | c100 traces protected: {n_c100}", flush=True)

grid = [(be, mn, b, r) for be in BACKENDS for mn in MODELS for b in BATCHES for r in RES]
for seed in SEEDS:
    order = grid[:]; random.Random(seed).shuffle(order)
    for be, mn, b, r in order:
        key = (str(seed), be, mn, str(b), str(r))
        if key in done: continue
        tag = f"s{seed} {be} {mn} b{b} r{r}"
        try:
            rr = one_run(seed, be, mn, b, r)
            row = [seed, be, mn, b, r, rr["status"], rr["ep"]] + [f"{rr[k]:.4f}" for k in COLS]
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: lat={rr['lat_med_ms']:.2f}ms "
                  f"E={rr['energy_mJ_per_inf']:.2f}mJ upd={rr['n_power_updates']}", flush=True)
        except torch.cuda.OutOfMemoryError:
            row = [seed, be, mn, b, r, "OOM", ""] + [""] * len(COLS)
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: OOM", flush=True); torch.cuda.empty_cache()
        except Exception as e:
            row = [seed, be, mn, b, r, "ERROR", str(e)[:100].replace(",", ";")] + [""] * len(COLS)
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: ERROR {str(e)[:120]}", flush=True)
        with open(CSV_OUT, "a", newline="") as f: csv.writer(f).writerow(row)
        torch.cuda.empty_cache()
print("rerun224 complete.", flush=True)
