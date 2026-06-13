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
    Returns the size of the model in MB by saving it to a temporary buffer.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    return size_mb
