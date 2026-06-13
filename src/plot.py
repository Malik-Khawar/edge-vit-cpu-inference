import matplotlib.pyplot as plt
import os

def generate_optimization_plots(results, save_dir="results"):
    """
    Generates diagnostic and benchmark plots for the EdgeViT optimizations.
    results: Dict containing metrics for each configuration:
             {
                 'Baseline (FP32)': {'latency': 114.40, 'size': 330.30, 'flops': 17.50, 'sim': 1.0},
                 'ToMe Only': {'latency': 85.20, 'size': 330.30, 'flops': 13.50, 'sim': 0.998},
                 ...
             }
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Bar Chart Comparison (Latency & Model Size)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    configs = list(results.keys())
    latencies = [results[c]['latency'] for c in configs]
    sizes = [results[c]['size'] for c in configs]
    
    x = range(len(configs))
    width = 0.35
    
    # Plot Latency
    color = '#2A7BDE'
    rects1 = ax1.bar([i - width/2 for i in x], latencies, width, label='Latency (ms)', color=color, alpha=0.85)
    ax1.set_xlabel('Model Configuration', fontweight='bold', labelpad=12)
    ax1.set_ylabel('Inference Latency (ms)', color=color, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15, ha='right')
    
    # Plot Model Size on secondary axis
    ax2 = ax1.twinx()
    color = '#E55934'
    rects2 = ax2.bar([i + width/2 for i in x], sizes, width, label='Model Size (MB)', color=color, alpha=0.85)
    ax2.set_ylabel('Model Size (MB)', color=color, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add values on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.1f}MB',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.title('EdgeViT Benchmark: Latency vs. Model Size', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    
    plot_path1 = os.path.join(save_dir, 'latency_size_comparison.png')
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"Saved bar chart comparison to {plot_path1}")
    
    # 2. Scatter Plot: Pareto Frontier of Latency vs. Feature Similarity
    plt.figure(figsize=(9, 6))
    
    similarities = [results[c]['sim'] * 100 for c in configs]
    
    # Use different colors for each point
    colors = ['#4A5568', '#E2E8F0', '#3182CE', '#DD6B20', '#38A169']
    if len(configs) > len(colors):
        colors = colors * (len(configs) // len(colors) + 1)
        
    plt.scatter(latencies, similarities, s=150, c=colors[:len(configs)], edgecolors='black', zorder=3)
    
    # Add labels and grid
    for i, txt in enumerate(configs):
        plt.annotate(txt, (latencies[i], similarities[i]), xytext=(8, -4), textcoords='offset points', fontweight='bold')
        
    plt.axhline(y=100.0, color='r', linestyle='--', alpha=0.5, label='Baseline Fidelity (100%)')
    
    plt.xlabel('CPU Inference Latency (ms, lower is better)', fontweight='bold', labelpad=10)
    plt.ylabel('Feature Cosine Similarity (%, higher is better)', fontweight='bold', labelpad=10)
    plt.title('EdgeViT Optimization Pareto Frontier: Speed vs. Fidelity', fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Adjust axes
    plt.xlim(min(latencies) - 10, max(latencies) + 15)
    plt.ylim(min(similarities) - 2, 102)
    
    plt.tight_layout()
    plot_path2 = os.path.join(save_dir, 'pareto_frontier.png')
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Saved Pareto Frontier plot to {plot_path2}")
    
    return plot_path1, plot_path2
