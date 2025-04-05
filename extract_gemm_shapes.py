import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import os

# 🧠 Pick device (MPS, CUDA, or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 📁 Local path to your Hugging Face-converted model
model_path = "/Users/f7b/.hf/Llama3.2-1B"

# ✅ Load tokenizer and model onto correct device
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32, 
    local_files_only=True)
model.to(device)
model.eval()

# 📊 GEMM shape tracker
gemm_shapes = defaultdict(int)

# 🔍 Hook for tracking GEMM shapes in Linear layers
def linear_hook(module, input, output):
    try:
        a = input[0]
        b = module.weight

        if a.dim() > 2:
            a = a.view(-1, a.size(-1))  # Flatten [B, S, K] → [M, K]

        M, K = a.shape
        N = b.shape[0]
        gemm_shapes[(M, N, K)] += 1
    except Exception as e:
        print(f"⚠️ Skipping layer due to error: {e}")

# Register hooks
hooks = []
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        hooks.append(module.register_forward_hook(linear_hook))

# 🧪 Dummy input for testing
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt").to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    _ = model(**inputs)

# Remove hooks
for h in hooks:
    h.remove()

# ✅ Print captured GEMM shapes
print("\n✅ Observed GEMM shapes (M, N, K):")
for shape, count in sorted(gemm_shapes.items(), key=lambda x: -x[1]):
    print(f"{shape} used {count} time(s)")