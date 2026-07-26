import torch
import torch.nn as nn
import os

def apply_dynamic_quantization(model):
    """
    Applies INT8 Dynamic Quantization to the Linear layers of the model.
    This reduces the weight memory footprint by ~4x and speeds up CPU inference.
    We use dynamic quantization because statically quantizing LayerNorm and Attention 
    softmaxes in Hugging Face models often degrades accuracy significantly.
    """
    print("Applying INT8 Dynamic Quantization to Linear layers...")
    
    # We quantize only the nn.Linear layers.
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized_model

def get_model_size_mb(model):
    """
    Returns the size of the model parameters and buffers in MB.
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb
