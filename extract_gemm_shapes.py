# This script extracts GEMM shapes from a Hugging Face model.
# It uses PyTorch to load the model and then iterates through its state dictionary
# to find tensors with two dimensions (which are typically used in GEMM operations).
# It then prints the shapes of these tensors, which can be useful for understanding
# the model's architecture and for optimizing performance.

##
#python convert_llama_weights_to_hf.py \
#    --input_dir ~/.llama/checkpoints/Llama3.2-1B 
#    --model_size 1B 
#    --llama_version 3.2 
#    --output_dir ~/.hf/Llama3.2-1B

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract GEMM shapes from a Hugging Face model by tracking Linear layer operations."
    )
    parser.add_argument(
        "model_path", 
        type=str, 
        help="Local path to the Hugging Face-converted model directory"
    )
    args = parser.parse_args()
    model_path = args.model_path

    # 🧠 Pick device (MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ✅ Load tokenizer and model onto correct device
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32, 
        local_files_only=True
    )
    model.to(device)
    model.eval()

    # 📊 GEMM shape tracker
    gemm_shapes = defaultdict(int)

    # 🔍 Hook for tracking GEMM shapes in Linear layers
    def linear_hook(module, input, output):
        try:
            a = input[0]
            b = module.weight

            # Flatten inputs with more than 2 dimensions: [B, S, K] → [M, K]
            if a.dim() > 2:
                a = a.view(-1, a.size(-1))

            M, K = a.shape
            N = b.shape[0]
            gemm_shapes[(M, N, K)] += 1
        except Exception as e:
            print(f"⚠️ Skipping layer due to error: {e}")

    # Register hooks on all Linear layers
    hooks = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    # 🧪 Dummy input for testing
    input_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(input_text, return_tensors="pt")
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

if __name__ == "__main__":
    main()