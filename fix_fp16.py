# Regenerate fp16 ONNX via ORT's transformers optimizer (ViT-aware types).
from pathlib import Path
ROOT = Path.home()/"projects"/"vit_backend_adaptive"
src = ROOT/"model"/"vit_tiny_patch16_224_cifar10_224.onnx"
dst = ROOT/"model"/"vit_tiny_patch16_224_cifar10_224_fp16.onnx"
if dst.exists(): dst.unlink()
from onnxruntime.transformers import optimizer
m = optimizer.optimize_model(str(src), model_type="vit", num_heads=3, hidden_size=192)
m.convert_float_to_float16(keep_io_types=True)
m.save_model_to_file(str(dst))
print("saved", dst)
import onnxruntime as ort
s = ort.InferenceSession(str(dst), providers=["CUDAExecutionProvider","CPUExecutionProvider"])
print("EP:", s.get_providers()[0])
