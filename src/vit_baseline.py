import torch
from transformers import ViTForImageClassification, ViTImageProcessor

def load_vit_baseline():
    """
    Loads the Hugging Face google/vit-base-patch16-224 model.
    """
    model_name = "google/vit-base-patch16-224"
    print(f"Loading baseline {model_name}...")
    
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
