### This script benchmarks the performance of various matrix multiplication operations


import time
import torch
import tabulate

# Custom benchmarking function to replace triton.testing.do_bench.
def synchronize_device(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and callable(torch.mps.synchronize):
        torch.mps.synchronize()

def do_bench(fn, warmup, rep, device):
    # Warm-up iterations.
    for _ in range(warmup):
        fn()
        synchronize_device(device)
    # Benchmark iterations.
    start = time.time()
    for _ in range(rep):
        fn()
        synchronize_device(device)
    end = time.time()
    total_time = end - start
    avg_time_ms = (total_time / rep) * 1000  # convert to milliseconds
    return avg_time_ms

torch.manual_seed(0)
repeats = 200
warmup = 30
timeout = 10

# Device selection:
# - Use MPS if available (Apple Silicon).
# - Otherwise, use CUDA which might be an NVIDIA or AMD GPU (via ROCm).
if torch.backends.mps.is_available():
    device = "mps"
    is_nvidia = False
    is_amd = False
    fp8_supported = False  # FP8 tests are not supported on MPS.
else:
    device = "cuda"
    device_name = torch.cuda.get_device_name(0).lower()
    is_nvidia = "nvidia" in device_name
    is_amd = "amd" in device_name
    # For CUDA devices:
    if is_nvidia:
        fp8_supported = True
    elif is_amd:
        # AMD MI250X does NOT support FP8; AMD300X and above do.
        if "mi250x" in device_name:
            fp8_supported = False
        else:
            fp8_supported = True
    else:
        fp8_supported = False  # default fallback

dtype_bf16 = torch.bfloat16

# FP8 types are only defined when on CUDA and FP8 is supported.
if device == "cuda" and fp8_supported:
    dtype_fp8_e5m2 = torch.float8_e5m2
    # For FP8 e4m3, use different dtypes depending on vendor.
    dtype_fp8_e4m3 = torch.float8_e4m3fn if is_nvidia else torch.float8_e4m3fnuz
else:
    dtype_fp8_e5m2 = None
    dtype_fp8_e4m3 = None

# GEMM Shapes: realistic sizes for benchmarking.
shapes = [
    (16384, 8192, 1280),
    (16384, 1024, 8192),
    (16384, 8192, 7168),
    (16384, 3584, 8192),
    (8192, 8192, 8192)
]

results = []

# FP8 Recipe setup: only applicable on CUDA and (typically) NVIDIA.
if device == "cuda":
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        import_te = True
    except ImportError:
        import_te = False
else:
    import_te = False

if device == "cuda" and import_te and is_nvidia:
    fp8_format = recipe.Format.HYBRID
    fp8_recipe = recipe.DelayedScaling(
        fp8_format=fp8_format,
        amax_history_len=16,
        amax_compute_algo="max"
    )

for (m, n, k) in shapes:
    # Calculate total FLOPS for the GEMM operation.
    nFLOPS = 2 * m * n * k

    # Benchmark using bf16 torch.matmul.
    a = torch.randn(m, k, device=device, dtype=dtype_bf16)
    b = torch.randn(n, k, device=device, dtype=dtype_bf16).transpose(-1, -2)
    with torch.inference_mode():
        ms_bf16 = do_bench(lambda: torch.matmul(a, b), warmup, repeats, device)
    tflops_bf16 = nFLOPS / ms_bf16 * 1e-9
    time.sleep(timeout)

    # Benchmark using FP8 TE.Linear (only if CUDA, transformer_engine is available, and NVIDIA).
    if device == "cuda" and import_te and is_nvidia:
        input_tensor = torch.randn(m, k, device=device)
        linear_layer = te.Linear(k, n, bias=False).to(device)
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            ms_te_linear = do_bench(lambda: linear_layer(input_tensor), warmup, repeats, device)
        tflops_te_linear = nFLOPS / ms_te_linear * 1e-9
        time.sleep(timeout)
    else:
        tflops_te_linear = 0.0

    # Benchmark FP8 torch._scaled_mm operations (only if running on CUDA and FP8 is supported).
    if device == "cuda" and fp8_supported:
        # FP8 e5m2 / e4m3fn branch.
        a_fp8_e5m2 = torch.randn(m, k, device=device).to(dtype_fp8_e5m2)
        b_fp8_e4m3 = torch.randn(n, k, device=device).to(dtype_fp8_e4m3).transpose(-1, -2)
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        with torch.inference_mode():
            try:
                # Attempt the 2-argument call first.
                ms_fp8_scaled_mm_e5m2 = do_bench(
                    lambda: torch._scaled_mm(a_fp8_e5m2, b_fp8_e4m3),
                    warmup, repeats, device)
            except TypeError:
                # Fallback to the 4-argument version if the PyTorch build supports it.
                ms_fp8_scaled_mm_e5m2 = do_bench(
                    lambda: torch._scaled_mm(a_fp8_e5m2, b_fp8_e4m3, scale_a, scale_b),
                    warmup, repeats, device
                )
        tflops_fp8_scaled_mm_e5m2 = nFLOPS / ms_fp8_scaled_mm_e5m2 * 1e-9
        time.sleep(timeout)

        # FP8 e4m3 branch.
        a_fp8_e4m3 = torch.randn(m, k, device=device).to(dtype_fp8_e4m3)
        b_fp8_e4m3 = torch.randn(n, k, device=device).to(dtype_fp8_e4m3).transpose(-1, -2)
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        with torch.inference_mode():
            try:
                ms_fp8_scaled_mm_e4m3 = do_bench(
                    lambda: torch._scaled_mm(a_fp8_e4m3, b_fp8_e4m3),
                    warmup, repeats, device
                )
            except Exception:
                ms_fp8_scaled_mm_e4m3 = do_bench(
                    lambda: torch._scaled_mm(a_fp8_e4m3, b_fp8_e4m3, scale_a, scale_b),
                    warmup, repeats, device
                )
        tflops_fp8_scaled_mm_e4m3 = nFLOPS / ms_fp8_scaled_mm_e4m3 * 1e-9
        time.sleep(timeout)
    else:
        tflops_fp8_scaled_mm_e5m2 = 0.0
        tflops_fp8_scaled_mm_e4m3 = 0.0

    # Collect results for this GEMM shape.
    results.append([
        f"({m}, {n}, {k})",
        f"{tflops_bf16:.1f} TFLOPS",
        f"{tflops_te_linear:.1f} TFLOPS",
        f"{tflops_fp8_scaled_mm_e5m2:.1f} TFLOPS",
        f"{tflops_fp8_scaled_mm_e4m3:.1f} TFLOPS"
    ])

# Print benchmark results.
headers = [
    "Shape (M, N, K)",
    "bf16 torch.matmul",
    "FP8 TE.Linear (autocast, bias=False)",
    "FP8 torch._scaled_mm (e5m2/e4m3fn)",
    "FP8 torch._scaled_mm (e4m3)"
]
print(f"Benchmark results for Realistic GEMM shapes with warmup={warmup} and repeats={repeats}")
print(tabulate.tabulate(results, headers=headers, tablefmt="grid"))
