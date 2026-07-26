import torch
from transformers import ViTForImageClassification, ViTImageProcessor

def load_vit_baseline(model_name="facebook/deit-tiny-patch16-224"):
    """
    Loads Hugging Face Vision Transformer model.
    Default: 'facebook/deit-tiny-patch16-224' (5.7M params, 1.2 GFLOPs) for fast CPU execution.
    Can also be set to 'google/vit-base-patch16-224' (86.6M params, 17.5 GFLOPs).
    """
    import os
    # Optimize CPU PyTorch threading to avoid OpenMP overhead
    num_threads = max(1, min(4, (os.cpu_count() or 4) // 2))
    torch.set_num_threads(num_threads)
    
    print(f"Loading baseline {model_name} (PyTorch CPU Threads: {num_threads})...")
    
    # We load the image processor and the model
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    
    # Set to eval mode for inference benchmarking
    model.eval()
    
    return model, processor

if __name__ == "__main__":
    model, processor = load_vit_baseline()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Dummy input test
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Output logits shape: {out.logits.shape}")
