import time
import torch
import psutil
import os
import numpy as np

def measure_latency(model, dummy_input, num_warmup=5, num_runs=20):
    """
    Measures the inference latency of the model on CPU.
    """
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    dummy_input = dummy_input.to(device)
    
    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            try:
                _ = model(dummy_input)
            except Exception as e:
                pass # Handle EarlyExitException if it occurs during warmup
            
    latencies = []
    with torch.inference_mode():
        for _ in range(num_runs):
            start = time.perf_counter()
            try:
                _ = model(dummy_input)
            except Exception as e:
                if type(e).__name__ != 'EarlyExitException':
                    raise e
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    return avg_latency, std_latency

def get_ram_usage():
    """
    Returns the current process RAM usage in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def estimate_flops(model, input_shape=(1, 3, 224, 224)):
    """
    Estimates FLOPs. A standard ViT-Base has ~17.5 GFLOPs.
    If ToMe and Early Exit are applied, we estimate the reduction.
    For demonstration, we return a mock value based on the model's attributes.
    """
    total_params = sum(p.numel() for p in model.parameters())
    base_gflops = 17.5 * (total_params / 86_600_000.0)
    
    # Check for early exit and ToMe
    layers = 12
    exit_layer = 12
    if hasattr(model, 'early_exit_head'):
        exit_layer = 6 # We put it at layer 5 (index 5) -> 6th layer
        
    layers_list = model.vit.layers if hasattr(model.vit, "layers") else model.vit.encoder.layer
    r_total = 0
    if hasattr(layers_list[0], '_tome_applied'):
        # For simplicity, assume r=4
        r_total = 4 * exit_layer
        
    # Roughly linear with layers and quadratic with sequence length
    # Original seq len = 196 + 1
    seq_len = 197
    
    # GFLOPs scaling approx
    gflops = base_gflops * (exit_layer / layers)
    
    if r_total > 0:
        # Very rough approximation of FLOPs reduction due to sequence shrinking
        avg_seq_len = seq_len - (r_total / 2)
        gflops = gflops * ((avg_seq_len / seq_len) ** 1.5)
        
    return gflops
