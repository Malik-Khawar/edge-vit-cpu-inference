import torch
import copy
import os
from src.data import get_imagenette_data
from src.vit_baseline import load_vit_baseline
from src.token_merging import apply_tome_to_model
from src.early_exit import EarlyExitHead, patch_encoder_for_early_exit, train_early_exit
from src.quantization import apply_dynamic_quantization, get_model_size_mb
from src.profile import measure_latency, get_ram_usage, estimate_flops
from src.evaluate import evaluate_feature_similarity
from src.plot import generate_optimization_plots

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
    
    # We will build results dict for plotting
    results = {
        'Baseline (FP32)': {
            'latency': base_latency,
            'size': base_size,
            'flops': base_gflops,
            'sim': 1.0
        }
    }
    
    # 3. Optimize: ToMe Only
    print("\n--- Benchmarking ToMe Only ---")
    tome_model = copy.deepcopy(baseline_model)
    apply_tome_to_model(tome_model, r_per_layer=4)
    tome_latency, _ = measure_latency(tome_model, dummy_input)
    tome_size = get_model_size_mb(tome_model)
    tome_gflops = estimate_flops(tome_model)
    tome_sim = evaluate_feature_similarity(baseline_model, tome_model, val_loader, device="cpu")
    print(f"ToMe Latency: {tome_latency:.2f} ms | Size: {tome_size:.2f} MB | Sim: {tome_sim:.2%}")
    results['ToMe Only'] = {
        'latency': tome_latency,
        'size': tome_size,
        'flops': tome_gflops,
        'sim': tome_sim
    }
    
    # 4. Train Early Exit Head (using ToMe model for feature extraction)
    exit_layer = 5 # 6th layer
    early_exit_head = EarlyExitHead(hidden_size=768, num_classes=1000)
    early_exit_head = train_early_exit(
        tome_model, early_exit_head, train_loader, val_loader, 
        exit_layer=exit_layer, epochs=1, device=device
    )
    
    # 5. Benchmarking Early Exit Only (No ToMe, No Quantization)
    print("\n--- Benchmarking Early Exit Only ---")
    ee_model = copy.deepcopy(baseline_model)
    patch_encoder_for_early_exit(ee_model, early_exit_head, exit_layer=exit_layer, entropy_threshold=1.5)
    ee_latency, _ = measure_latency(ee_model, dummy_input)
    ee_size = get_model_size_mb(ee_model)
    ee_gflops = estimate_flops(ee_model)
    ee_sim = evaluate_feature_similarity(baseline_model, ee_model, val_loader, device="cpu")
    print(f"Early Exit Latency: {ee_latency:.2f} ms | Size: {ee_size:.2f} MB | Sim: {ee_sim:.2%}")
    results['Early Exit Only'] = {
        'latency': ee_latency,
        'size': ee_size,
        'flops': ee_gflops,
        'sim': ee_sim
    }
    
    # 6. Benchmarking Quantized Only (No ToMe, No Early Exit)
    print("\n--- Benchmarking Quantized Only ---")
    quant_model = copy.deepcopy(baseline_model)
    quant_model.to("cpu")
    quant_model = apply_dynamic_quantization(quant_model)
    quant_latency, _ = measure_latency(quant_model, dummy_input)
    quant_size = get_model_size_mb(quant_model)
    quant_gflops = estimate_flops(quant_model)
    quant_sim = evaluate_feature_similarity(baseline_model, quant_model, val_loader, device="cpu")
    print(f"Quantized Latency: {quant_latency:.2f} ms | Size: {quant_size:.2f} MB | Sim: {quant_sim:.2%}")
    results['Quantized Only'] = {
        'latency': quant_latency,
        'size': quant_size,
        'flops': quant_gflops,
        'sim': quant_sim
    }
    
    # 7. Benchmarking EdgeViT (Full Optimization: ToMe + Early Exit + Quantization)
    print("\n--- Benchmarking EdgeViT (Combined) ---")
    edge_model = copy.deepcopy(tome_model) # Has ToMe
    patch_encoder_for_early_exit(edge_model, early_exit_head, exit_layer=exit_layer, entropy_threshold=1.5) # Has Early Exit
    edge_model.to("cpu")
    edge_model = apply_dynamic_quantization(edge_model) # Has Quantization
    
    edge_latency, _ = measure_latency(edge_model, dummy_input)
    edge_size = get_model_size_mb(edge_model)
    edge_gflops = estimate_flops(edge_model)
    edge_sim = evaluate_feature_similarity(baseline_model, edge_model, val_loader, device="cpu")
    print(f"EdgeViT Latency: {edge_latency:.2f} ms | Size: {edge_size:.2f} MB | Sim: {edge_sim:.2%}")
    results['EdgeViT (Full)'] = {
        'latency': edge_latency,
        'size': edge_size,
        'flops': edge_gflops,
        'sim': edge_sim
    }
    
    # 8. Generate Benchmark Plots
    print("\n--- Generating Benchmark Plots ---")
    generate_optimization_plots(results, save_dir="results")
    
    # 9. Summary Report
    print("\n================================================================================")
    print("Optimization Summary:")
    print("================================================================================")
    print(f"Configuration    | Latency (ms)  | Model Size (MB) | FLOPs (GFLOPs) | Feature Sim")
    print(f"--------------------------------------------------------------------------------")
    for name, metrics in results.items():
        print(f"{name:16s} | {metrics['latency']:12.2f}  | {metrics['size']:15.2f} | {metrics['flops']:14.2f} | {metrics['sim']:11.2%}")
    print("================================================================================")

if __name__ == "__main__":
    run_pipeline()
