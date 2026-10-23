# Cell 8 — static benchmark grid on GPU.
# 7 backends x 4 models x {1,8,32} x {224,384} x 3 seeds = 504 runs.
# Per run: 50 warmup + 200 timed latencies; separate >=10 s energy window
# (NVML @10 ms, counter updates ~100 ms on Ada); SM clocks logged; memory =
# total-used minus idle baseline. Resume-safe via output CSV. OOM recorded.
import os, sys, site, glob, time, csv, json, random, threading, itertools
from pathlib import Path

ROOT = Path.home()/"projects"/"vit_backend_adaptive"
DIRS = {k: ROOT/k for k in ["dataset","output","model","logs"]}

sp = site.getsitepackages()[0]
import ctypes
for d in sorted(glob.glob(f"{sp}/nvidia/*/lib")) + [f"{sp}/tensorrt_libs"]:
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if ".so" in f and "nvblas" not in f:
                try: ctypes.CDLL(os.path.join(d,f), mode=ctypes.RTLD_GLOBAL)
                except OSError: pass
cc = next(iter(glob.glob(f"{sys.prefix}/bin/*-conda-linux-gnu-gcc")), None)
if cc: os.environ["CC"]=cc; os.environ["CXX"]=cc.replace("gcc","g++")

import numpy as np, torch, timm
import onnxruntime as ort
import pynvml

HW_ID = torch.cuda.get_device_name(0).replace("NVIDIA ","").replace(" ","_")
SEEDS=[42,1234,2024]
MODELS = {"vit_tiny_patch16_224":(3,192),"vit_small_patch16_224":(6,384),
          "vit_base_patch16_224":(12,768),"deit_small_patch16_224":(6,384)}
BACKENDS = ["eager","torchscript","inductor_default","inductor_maxautotune",
            "ort_cuda_fp32","ort_cuda_fp16","ort_trt"]
BATCHES=[1,8,32]; RESOLUTIONS=[224,384]
N_WARM, N_TIMED = 50, 200
ENERGY_MIN_S = 10.0
FPS_TARGETS=[30,60,120]
DS_FOR_CKPT = "cifar10"   # grid measures compute, one checkpoint per model suffices
N_CLASSES = 10
TRT_CACHE = DIRS["model"]/"trt_cache"; TRT_CACHE.mkdir(exist_ok=True)

pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)
def gpu_mem(): return pynvml.nvmlDeviceGetMemoryInfo(H).used/1024**2
def sm_clock(): return pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)
def power_w(): return pynvml.nvmlDeviceGetPowerUsage(H)/1000.0

def load_ft(mn, res=224):
    m = timm.create_model(mn,pretrained=False,num_classes=N_CLASSES,img_size=res)
    sd = torch.load(DIRS["model"]/f"{mn}_{DS_FOR_CKPT}_best.pth",
                    map_location="cpu",weights_only=True)
    if res != 224:
        from timm.models.vision_transformer import checkpoint_filter_fn
        sd = checkpoint_filter_fn(sd, m)   # pos-embed interpolation 224->res
    m.load_state_dict(sd)
    m.eval(); return m

def onnx_path(mn,res,fp16=False):
    p = DIRS["model"]/f"{mn}_{DS_FOR_CKPT}_{res}.onnx"
    if not p.exists():
        torch.onnx.export(load_ft(mn,res), torch.randn(1,3,res,res), p,
            input_names=["input"],output_names=["logits"],
            dynamic_axes={"input":{0:"batch"},"logits":{0:"batch"}},opset_version=17)
    if not fp16: return p
    q = DIRS["model"]/f"{mn}_{DS_FOR_CKPT}_{res}_fp16.onnx"
    if not q.exists():
        from onnxruntime.transformers import optimizer
        nh,hs = MODELS[mn]
        om = optimizer.optimize_model(str(p),model_type="vit",num_heads=nh,hidden_size=hs)
        om.convert_float_to_float16(keep_io_types=True)
        tmp=q.with_suffix(".tmp"); om.save_model_to_file(str(tmp)); Path(tmp).rename(q)
    return q

def make_infer(be,mn,res):
    if be in ("eager","torchscript","inductor_default","inductor_maxautotune"):
        m = load_ft(mn,res).cuda()
        if be=="torchscript": m=torch.jit.script(m)
        elif be.startswith("inductor"):
            m=torch.compile(m,mode="max-autotune" if be.endswith("maxautotune") else "default",
                            dynamic=False)
        @torch.no_grad()
        def infer(x): return m(x)
        return infer,"torch"
    p = onnx_path(mn,res,fp16=(be=="ort_cuda_fp16"))
    if be=="ort_trt":
        prov=[("TensorrtExecutionProvider",{"trt_engine_cache_enable":True,
              "trt_engine_cache_path":str(TRT_CACHE),"trt_fp16_enable":True,
              "trt_timing_cache_enable":True}),"CUDAExecutionProvider","CPUExecutionProvider"]
        want="TensorrtExecutionProvider"
    else:
        prov=["CUDAExecutionProvider","CPUExecutionProvider"]; want="CUDAExecutionProvider"
    sess=ort.InferenceSession(str(p), ort.SessionOptions(), providers=prov)
    if sess.get_providers()[0]!=want:
        raise RuntimeError(f"{be}: EP fallback to {sess.get_providers()[0]}")
    def infer(x):
        return torch.from_numpy(sess.run(["logits"],{"input":x.cpu().numpy()})[0]).cuda()
    return infer,want

OUT = DIRS["output"]/f"static_grid_{HW_ID}.csv"
RAW = DIRS["output"]/f"static_grid_raw_{HW_ID}"; RAW.mkdir(exist_ok=True)
if not OUT.exists():
    OUT.write_text("seed,backend,model,batch,res,status,ep,"
        "lat_med_ms,lat_p5_ms,lat_p95_ms,lat_mean_ms,"
        "energy_mJ_per_inf,power_mean_W,n_power_updates,energy_win_s,"
        "mem_MB,idle_mem_MB,sm_clock_mean,sm_clock_min,"
        "miss30,miss60,miss120,setup_s\n")
done=set()
with open(OUT) as f:
    for r in csv.DictReader(f): done.add((r["seed"],r["backend"],r["model"],r["batch"],r["res"]))

def one_run(seed,be,mn,b,res):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.empty_cache(); time.sleep(1.0)
    idle_mem = gpu_mem()
    t_setup0=time.perf_counter()
    infer,ep = make_infer(be,mn,res)
    x = torch.randn(b,3,res,res,device="cuda")
    with torch.no_grad(): _=infer(x)
    torch.cuda.synchronize()
    setup_s=time.perf_counter()-t_setup0
    # --- latency window ---
    with torch.no_grad():
        for _ in range(N_WARM): infer(x)
        torch.cuda.synchronize()
        lats=[]; clocks=[]
        for i in range(N_TIMED):
            t0=time.perf_counter(); infer(x); torch.cuda.synchronize()
            lats.append((time.perf_counter()-t0)*1000)
            if i%20==0: clocks.append(sm_clock())
    lats=np.array(lats)
    mem = gpu_mem()
    # --- energy window (>=10 s continuous) ---
    samples=[]; stop=threading.Event()
    def sampler():
        while not stop.is_set():
            samples.append((time.perf_counter(), power_w())); time.sleep(0.01)
    th=threading.Thread(target=sampler); th.start()
    n_inf=0; t0=time.perf_counter()
    with torch.no_grad():
        while time.perf_counter()-t0 < ENERGY_MIN_S:
            infer(x); n_inf+=1
        torch.cuda.synchronize()
    t1=time.perf_counter(); stop.set(); th.join()
    ts=np.array([s[0] for s in samples]); ps=np.array([s[1] for s in samples])
    mask=(ts>=t0)&(ts<=t1)
    E_J=np.trapezoid(ps[mask],ts[mask]) if mask.sum()>1 else float("nan")
    n_upd=int((np.diff(ps[mask])!=0).sum()) if mask.sum()>1 else 0
    e_mj = E_J*1000/max(n_inf,1)/b          # per single inference
    miss={f:float((lats>1000.0/f).mean()) for f in FPS_TARGETS}
    np.savez(RAW/f"{seed}_{be}_{mn}_{b}_{res}.npz", lats=lats,
             power_t=ts[mask], power_w=ps[mask], clocks=np.array(clocks))
    return dict(status="ok",ep=ep,
        lat_med_ms=np.median(lats),lat_p5_ms=np.percentile(lats,5),
        lat_p95_ms=np.percentile(lats,95),lat_mean_ms=lats.mean(),
        energy_mJ_per_inf=e_mj,power_mean_W=ps[mask].mean() if mask.sum() else float("nan"),
        n_power_updates=n_upd,energy_win_s=t1-t0,
        mem_MB=mem-idle_mem,idle_mem_MB=idle_mem,
        sm_clock_mean=float(np.mean(clocks)),sm_clock_min=float(np.min(clocks)),
        miss30=miss[30],miss60=miss[60],miss120=miss[120],setup_s=setup_s)

grid=[(be,mn,b,res) for be in BACKENDS for mn in MODELS
      for b in BATCHES for res in RESOLUTIONS]
for seed in SEEDS:
    order=grid[:]; random.Random(seed).shuffle(order)   # per-seed order randomization
    for be,mn,b,res in order:
        key=(str(seed),be,mn,str(b),str(res))
        if key in done: continue
        tag=f"s{seed} {be} {mn} b{b} r{res}"
        try:
            r=one_run(seed,be,mn,b,res)
            row=[seed,be,mn,b,res,r["status"],r["ep"]]+[f"{r[k]:.4f}" if isinstance(r[k],float) else r[k]
                 for k in ["lat_med_ms","lat_p5_ms","lat_p95_ms","lat_mean_ms",
                           "energy_mJ_per_inf","power_mean_W","n_power_updates","energy_win_s",
                           "mem_MB","idle_mem_MB","sm_clock_mean","sm_clock_min",
                           "miss30","miss60","miss120","setup_s"]]
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: lat={r['lat_med_ms']:.2f}ms "
                  f"E={r['energy_mJ_per_inf']:.2f}mJ upd={r['n_power_updates']}", flush=True)
        except torch.cuda.OutOfMemoryError:
            row=[seed,be,mn,b,res,"OOM",""]+[""]*16
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: OOM", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            row=[seed,be,mn,b,res,"ERROR",str(e)[:100]]+[""]*16
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: ERROR {str(e)[:120]}", flush=True)
        with open(OUT,"a") as f: csv.writer(f).writerow(row)
        torch.cuda.empty_cache()
print("Cell 8 complete.", flush=True)
pynvml.nvmlShutdown()
