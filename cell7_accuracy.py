# Cell 7 — accuracy equivalence: 12 checkpoints x 7 backends.
# Outputs: output/accuracy_equivalence_{HW}.csv  (model,dataset,backend,top1,maxdiff,ep,seconds)
# Resume-safe: existing (model,dataset,backend) rows are skipped.
import os, sys, site, glob, time, csv, random
from pathlib import Path

ROOT = Path.home()/"projects"/"vit_backend_adaptive"
DIRS = {k: ROOT/k for k in ["dataset","output","model","logs"]}

# --- preload pip-installed CUDA/TensorRT libraries ---
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

import numpy as np, torch, timm, torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import onnxruntime as ort

HW_ID = torch.cuda.get_device_name(0).replace("NVIDIA ","").replace(" ","_")
SEEDS=[42,1234,2024]
random.seed(SEEDS[0]); np.random.seed(SEEDS[0]); torch.manual_seed(SEEDS[0])

MODELS = {  # name -> (num_heads, hidden_size) for ORT fp16 optimizer
    "vit_tiny_patch16_224": (3,192), "vit_small_patch16_224": (6,384),
    "vit_base_patch16_224": (12,768), "deit_small_patch16_224": (6,384),
}
DATASETS = ["cifar10","cifar100","imagenette"]
BACKENDS = ["eager","torchscript","inductor_default","inductor_maxautotune",
            "ort_cuda_fp32","ort_cuda_fp16","ort_trt"]
N_CLASSES = {"cifar10":10,"cifar100":100,"imagenette":10}
NORM = {"cifar10":([0.4914,0.4822,0.4465],[0.2470,0.2435,0.2616]),
        "cifar100":([0.5071,0.4865,0.4409],[0.2673,0.2564,0.2762]),
        "imagenette":([0.485,0.456,0.406],[0.229,0.224,0.225])}
RES, BS = 224, 64
TRT_CACHE = DIRS["model"]/"trt_cache"; TRT_CACHE.mkdir(exist_ok=True)

def test_loader(ds):
    mean,std = NORM[ds]
    tf = T.Compose([T.Resize(256), T.CenterCrop(RES), T.ToTensor(), T.Normalize(mean,std)])
    if ds=="cifar10":  d=torchvision.datasets.CIFAR10(DIRS["dataset"],train=False,transform=tf)
    elif ds=="cifar100": d=torchvision.datasets.CIFAR100(DIRS["dataset"],train=False,transform=tf)
    else: d=torchvision.datasets.ImageFolder(DIRS["dataset"]/"imagenette2"/"val",transform=tf)
    return DataLoader(d,batch_size=BS,shuffle=False,num_workers=8,pin_memory=True)

def load_ft(mn,ds):
    m = timm.create_model(mn,pretrained=False,num_classes=N_CLASSES[ds])
    m.load_state_dict(torch.load(DIRS["model"]/f"{mn}_{ds}_best.pth",
                                 map_location="cpu",weights_only=True))
    m.eval(); return m

def onnx_fp32(mn,ds):
    p = DIRS["model"]/f"{mn}_{ds}_{RES}.onnx"
    if not p.exists():
        torch.onnx.export(load_ft(mn,ds), torch.randn(1,3,RES,RES), p,
            input_names=["input"],output_names=["logits"],
            dynamic_axes={"input":{0:"batch"},"logits":{0:"batch"}},opset_version=17)
    return p

def onnx_fp16(mn,ds):
    q = DIRS["model"]/f"{mn}_{ds}_{RES}_fp16.onnx"
    if not q.exists():
        from onnxruntime.transformers import optimizer
        nh,hs = MODELS[mn]
        m = optimizer.optimize_model(str(onnx_fp32(mn,ds)),model_type="vit",
                                     num_heads=nh,hidden_size=hs)
        m.convert_float_to_float16(keep_io_types=True)
        tmp = q.with_suffix(".tmp"); m.save_model_to_file(str(tmp)); Path(tmp).rename(q)
    return q

def make_infer(be,mn,ds):
    """returns (infer(gpu tensor)->gpu logits, ep_str)"""
    if be in ("eager","torchscript","inductor_default","inductor_maxautotune"):
        m = load_ft(mn,ds).cuda()
        if be=="torchscript": m = torch.jit.script(m)
        elif be.startswith("inductor"):
            mode = "max-autotune" if be.endswith("maxautotune") else "default"
            m = torch.compile(m,mode=mode,dynamic=False)
        @torch.no_grad()
        def infer(x): return m(x)
        return infer, "torch"
    p = onnx_fp16(mn,ds) if be=="ort_cuda_fp16" else onnx_fp32(mn,ds)
    if be=="ort_trt":
        prov=[("TensorrtExecutionProvider",{"trt_engine_cache_enable":True,
              "trt_engine_cache_path":str(TRT_CACHE),"trt_fp16_enable":True,
              "trt_timing_cache_enable":True}),"CUDAExecutionProvider","CPUExecutionProvider"]
        want="TensorrtExecutionProvider"
    else:
        prov=["CUDAExecutionProvider","CPUExecutionProvider"]; want="CUDAExecutionProvider"
    sess = ort.InferenceSession(str(p), ort.SessionOptions(), providers=prov)
    ep = sess.get_providers()[0]
    if ep != want: raise RuntimeError(f"{be}: wanted {want}, active {ep}")
    def infer(x):
        return torch.from_numpy(sess.run(["logits"],{"input":x.cpu().numpy()})[0]).cuda()
    return infer, ep

OUT = DIRS["output"]/f"accuracy_equivalence_{HW_ID}.csv"
if not OUT.exists():
    OUT.write_text("model,dataset,backend,top1,maxdiff_vs_eager,ep,seconds\n")
done = set()
with open(OUT) as f:
    for row in csv.DictReader(f):
        done.add((row["model"],row["dataset"],row["backend"]))

for mn in MODELS:
    for ds in DATASETS:
        loader = None; probe = None; ref_logits = None
        for be in BACKENDS:
            key=(mn,ds,be)
            if key in done:
                print(f"skip {key}"); continue
            t0=time.time()
            try:
                if loader is None:
                    loader = test_loader(ds)
                    probe,_ = next(iter(loader)); probe = probe[:8].cuda()
                infer, ep = make_infer(be,mn,ds)
                with torch.no_grad():
                    logits = infer(probe).float().cpu()
                if be=="eager": ref_logits = logits
                if ref_logits is None:  # resume case: rebuild eager ref quickly
                    e_inf,_ = make_infer("eager",mn,ds)
                    with torch.no_grad(): ref_logits = e_inf(probe).float().cpu()
                maxdiff = (logits-ref_logits).abs().max().item()
                correct=total=0
                with torch.no_grad():
                    for x,y in loader:
                        x=x.cuda(non_blocking=True)
                        pred = infer(x).argmax(1).cpu()
                        correct += (pred==y).sum().item(); total += y.numel()
                top1 = 100*correct/total
                secs = time.time()-t0
                with open(OUT,"a") as f:
                    csv.writer(f).writerow([mn,ds,be,f"{top1:.2f}",f"{maxdiff:.5f}",ep,f"{secs:.0f}"])
                print(f"[{time.strftime('%H:%M:%S')}] {mn}/{ds}/{be}: top1={top1:.2f} "
                      f"maxdiff={maxdiff:.5f} ep={ep} ({secs:.0f}s)", flush=True)
            except Exception as e:
                with open(OUT,"a") as f:
                    csv.writer(f).writerow([mn,ds,be,"ERROR","",str(e)[:120],""])
                print(f"[{time.strftime('%H:%M:%S')}] {mn}/{ds}/{be} FAILED: {str(e)[:150]}", flush=True)
            finally:
                torch.cuda.empty_cache()
print("Cell 7 complete.", flush=True)
