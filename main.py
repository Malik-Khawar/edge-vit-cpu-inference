import torch
import copy
from src.data import get_imagenette_data
from src.vit_baseline import load_vit_baseline
from src.token_merging import apply_tome_to_model
from src.early_exit import EarlyExitHead, patch_encoder_for_early_exit, train_early_exit
from src.quantization import apply_dynamic_quantization, get_model_size_mb
from src.profile import measure_latency, get_ram_usage, estimate_flops
from src.evaluate import evaluate_feature_similarity

def run_pipeline():
    print("================================================================================")
    print("Project 5: EdgeViT - High-Throughput CPU Inference for Vision Transformers")
    print("================================================================================")
    
    # 1. Load Data
    train_loader, val_loader = get_imagenette_data(batch_size=16) # Smaller batch for fast demo
    
    # We will use GPU for training the early exit head if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Load Baseline Model
    baseline_model, processor = load_vit_baseline()
    
    # Measure Baseline stats
    print("\n--- Benchmarking Baseline ViT ---")
    dummy_input = torch.randn(1, 3, 224, 224)
    base_latency, _ = measure_latency(baseline_model, dummy_input)
    base_size = get_model_size_mb(baseline_model)
    base_gflops = estimate_flops(baseline_model)
    
    print(f"Baseline Latency: {base_latency:.2f} ms")
    print(f"Baseline Size: {base_size:.2f} MB")
    print(f"Baseline FLOPs: {base_gflops:.2f} GFLOPs")
    
    # 3. Optimize Model
    print("\n--- Optimizing ViT for Edge Inference ---")
    optimized_model = copy.deepcopy(baseline_model)
    
    # Apply Token Merging (ToMe)
    r_per_layer = 4 # Merge 4 tokens per layer
    apply_tome_to_model(optimized_model, r_per_layer=r_per_layer)
    
    # Add Early Exit Head
    exit_layer = 5 # 6th layer
    early_exit_head = EarlyExitHead(hidden_size=768, num_classes=1000) # Imagenet 1000 classes because ViT is pre-trained on it
    
    # Train Early Exit Head (GPU if available)
    early_exit_head = train_early_exit(
        optimized_model, early_exit_head, train_loader, val_loader, 
        exit_layer=exit_layer, epochs=1, device=device
    )
    
    # Patch the encoder for inference routing
    patch_encoder_for_early_exit(optimized_model, early_exit_head, exit_layer=exit_layer, entropy_threshold=1.5)
    
    # Apply Dynamic INT8 Quantization
    optimized_model.to("cpu") # Must be on CPU for quantization
    optimized_model = apply_dynamic_quantization(optimized_model)
    
    # 4. Measure Optimized stats
    print("\n--- Benchmarking Optimized EdgeViT ---")
    opt_latency, _ = measure_latency(optimized_model, dummy_input)
    opt_size = get_model_size_mb(optimized_model)
    opt_gflops = estimate_flops(optimized_model)
    
    print(f"Optimized Latency: {opt_latency:.2f} ms")
    print(f"Optimized Size: {opt_size:.2f} MB")
    print(f"Optimized FLOPs: {opt_gflops:.2f} GFLOPs")
    
    opt_sim = evaluate_feature_similarity(baseline_model, optimized_model, val_loader, device="cpu")
    print(f"Feature Cosine Similarity: {opt_sim:.2%}")
    
    # 5. Summary Report
    print("\n================================================================================")
    print("Optimization Summary:")
    print("================================================================================")
    print(f"Metric       | Baseline       | EdgeViT Optimized | Improvement")
    print(f"--------------------------------------------------------------------------------")
    print(f"Latency      | {base_latency:8.2f} ms | {opt_latency:8.2f} ms     | {base_latency/opt_latency:.1f}x faster")
    print(f"Model Size   | {base_size:8.2f} MB | {opt_size:8.2f} MB     | {base_size/opt_size:.1f}x smaller")
    print(f"Compute      | {base_gflops:8.2f} GFL  | {opt_gflops:8.2f} GFL      | {base_gflops/opt_gflops:.1f}x fewer FLOPs")
    print(f"Feature Sim  | 100.00%        | {opt_sim:8.2%}       | Cosine Sim of embeddings")
    print("================================================================================")

if __name__ == "__main__":
    run_pipeline()
