# EdgeViT: High-Throughput CPU Inference for Vision Transformers

This project focuses on memory and inference speed optimizations for Vision Transformers (`google/vit-base-patch16-224`) on Edge/CPU environments.

## Techniques Implemented
- Token Merging (ToMe)
- Early Exits (Entropy-based)
- INT8 Post-Training Static Quantization (PTSQ)
