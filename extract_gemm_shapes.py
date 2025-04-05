# This script extracts the shapes of GEMM operations from a PyTorch model's state dictionary
# and classifies them based on their intended use (e.g., QKV projection, MLP, etc.).
# It uses heuristics based on parameter names to classify the roles of the parameters.
# The script then aggregates the shapes by their roles and prints the results.


#!/usr/bin/env python
import torch
from transformers import AutoModelForCausalLM  # or use AutoModel if more appropriate
from collections import defaultdict, Counter

def classify_layer(name: str) -> str:
    """
    Classify the role of a parameter based on its name.
    This heuristic looks for common substrings used in transformer models.
    """
    lower_name = name.lower()
    if any(token in lower_name for token in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
        return "QKV Projection"
    elif "attn" in lower_name and "proj" in lower_name:
        return "Attention Projection"
    elif any(token in lower_name for token in ["mlp", "fc", "dense", "ffn"]):
        return "MLP"
    elif "emb" in lower_name:
        return "Embedding"
    elif "lm_head" in lower_name or "output" in lower_name:
        return "Output"
    else:
        return "Other"

def extract_and_aggregate_gemm_shapes(model):
    """
    Extracts GEMM shapes from the model's state dictionary.
    For each 2D parameter (typically corresponding to linear layers used in GEMM operations),
    classify its intended use (e.g. QKV Projection, MLP, etc.) and tally the shape counts.
    
    Returns:
        A dict mapping each role (e.g. "QKV Projection", "MLP") to a Counter
        that tallies the occurrences of each unique shape.
    """
    role_shapes = defaultdict(Counter)
    for name, param in model.state_dict().items():
        if param.ndim == 2:  # Likely used in GEMM operations (linear layers)
            role = classify_layer(name)
            role_shapes[role][tuple(param.shape)] += 1
    return role_shapes

def main():
    # Path to your locally downloaded model
    model_path = "/Users/f7b/.hf/Llama3.2-1B"
    
    # Load the model; if your model is not a causal language model, consider using AutoModel instead
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Extract and aggregate GEMM shapes by role
    aggregated_shapes = extract_and_aggregate_gemm_shapes(model)
    
    # Print out the aggregated results with their potential roles
    print("Aggregated GEMM shapes by inferred role:")
    for role, shapes_counter in aggregated_shapes.items():
        print(f"\nRole: {role}")
        for shape, count in shapes_counter.items():
            print(f"  Shape {shape}: {count} occurrence(s)")

if __name__ == '__main__':
    main()
