import torch
import torch.nn.functional as F

def bipartite_soft_matching(metric: torch.Tensor, r: int):
    """
    Finds the top `r` most similar pairs in the token sequence.
    `metric` is usually the tokens themselves or the Keys from attention (B, N, C).
    We split N tokens into two sets: A (evens) and B (odds) to form a bipartite graph.
    """
    B, N, C = metric.shape
    if r <= 0:
        return None

    # We protect the first token ([CLS] token) from being merged!
    # Let's split the patches (excluding CLS)
    patches = metric[:, 1:, :] # (B, N-1, C)
    N_patches = N - 1

    # Split into sets A and B
    num_a = N_patches // 2
    
    a = patches[:, :num_a, :]
    b = patches[:, num_a:, :]

    # Compute cosine similarity
    # Normalize
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)

    # Cosine similarity matrix (B, num_a, num_b)
    scores = a_norm @ b_norm.transpose(-1, -2)

    # For each node in A, find the most similar node in B
    node_max, node_idx = scores.max(dim=-1) # node_max: (B, num_a), node_idx: (B, num_a)
    
    # Pick the top `r` edges across the batch
    # We sort the max scores
    top_scores, top_idx = node_max.sort(dim=-1, descending=True) # (B, num_a)
    
    # Select top r
    top_idx = top_idx[:, :r] # (B, r)
    
    # Now we have the indices in A (top_idx) and the corresponding indices in B (node_idx matched)
    # To apply the merge, we need a merging function
    # For simplicity, we return the indices to merge
    
    return top_idx, node_idx

def merge_tokens(x: torch.Tensor, top_idx: torch.Tensor, node_idx: torch.Tensor, r: int):
    """
    x: (B, N, C)
    Returns: x_merged (B, N-r, C)
    """
    B, N, C = x.shape
    
    # Separate CLS and patches
    cls_token = x[:, 0:1, :]
    patches = x[:, 1:, :]
    
    num_a = (N - 1) // 2
    num_b = (N - 1) - num_a
    
    a = patches[:, :num_a, :]
    b = patches[:, num_a:, :]
    
    # We want to merge the selected tokens from a into b.
    # We will literally average them.
    for batch_idx in range(B):
        # The indices in A that we selected
        a_indices = top_idx[batch_idx] # (r,)
        # The corresponding indices in B
        b_indices = node_idx[batch_idx][a_indices] # (r,)
        
        # Add the values of A to B
        b[batch_idx, b_indices, :] = (b[batch_idx, b_indices, :] + a[batch_idx, a_indices, :]) / 2.0
    
    # Now we need to drop the merged tokens from A
    # Create a mask for A
    mask_a = torch.ones((B, num_a), dtype=torch.bool, device=x.device)
    for batch_idx in range(B):
        mask_a[batch_idx, top_idx[batch_idx]] = False
        
    # We collect the remaining tokens
    # Because mask_a might have different number of True per batch if r varied? 
    # Here r is constant, so mask_a has exactly (num_a - r) Trues.
    a_remaining = []
    for batch_idx in range(B):
        a_remaining.append(a[batch_idx][mask_a[batch_idx]])
        
    a_remaining = torch.stack(a_remaining, dim=0) # (B, num_a - r, C)
    
    # Concatenate back
    new_patches = torch.cat([a_remaining, b], dim=1) # (B, N-1-r, C)
    new_x = torch.cat([cls_token, new_patches], dim=1) # (B, N-r, C)
    
    return new_x

def apply_tome_to_layer(layer, r: int):
    """
    Patches a Hugging Face ViTLayer to apply ToMe at the end of the layer.
    """
    original_forward = layer.forward
    
    def new_forward(*args, **kwargs):
        # Run original layer
        outputs = original_forward(*args, **kwargs)
        
        # Outputs is a tuple (hidden_states, attentions) or a tensor
        x = outputs[0] if isinstance(outputs, tuple) else outputs
        
        if r > 0 and x.shape[1] > 1 + r:
            # Apply token merging
            matching = bipartite_soft_matching(x, r)
            if matching is not None:
                top_idx, node_idx = matching
                x = merge_tokens(x, top_idx, node_idx, r)
                
            # Replace hidden_states in the output
            if isinstance(outputs, tuple):
                outputs = (x,) + outputs[1:]
            else:
                outputs = x
            
        return outputs

    # Bind the new method
    layer.forward = new_forward
    layer._tome_applied = True

def apply_tome_to_model(model, r_per_layer: int):
    """
    Applies Token Merging to all layers of a Hugging Face ViT model.
    """
    # ViT-Base has 12 layers. We apply ToMe to all layers.
    # Total tokens removed = 12 * r_per_layer
    layers = model.vit.layers if hasattr(model.vit, "layers") else model.vit.encoder.layer
    for i, layer in enumerate(layers):
        apply_tome_to_layer(layer, r_per_layer)
    print(f"Applied Token Merging to all 12 layers (r={r_per_layer} per layer).")

