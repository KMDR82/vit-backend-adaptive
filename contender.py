# S3 contention load generator: duty-cycled FP16 matmul in its own process.
# Usage: python contender.py <duty 0..100> [n]   (runs until killed)
import sys, time
import torch

duty = float(sys.argv[1]) / 100.0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
period = 0.1  # 100 ms cycle
a = torch.randn(n, n, device="cuda", dtype=torch.float16)
b = torch.randn(n, n, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
print(f"contender up: duty={duty:.0%} n={n}", flush=True)
while True:
    t0 = time.perf_counter()
    if duty > 0:
        while time.perf_counter() - t0 < period * duty:
            c = a @ b
            torch.cuda.synchronize()   # sync per matmul: wall-time duty = real GPU duty
    rest = period - (time.perf_counter() - t0)
    if rest > 0:
        time.sleep(rest)
