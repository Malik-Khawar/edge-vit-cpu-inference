import torch
import torch.nn as nn
from tqdm import tqdm

class EarlyExitHead(nn.Module):
    def __init__(self, hidden_size=768, num_classes=10):
        super().__init__()
        # Simple linear head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x is (B, N, C)
        # ViT typically uses the [CLS] token (index 0) for classification
        cls_token = x[:, 0, :]
        return self.classifier(cls_token)

class EarlyExitException(Exception):
    """Exception to break out of the transformer loop early."""
    def __init__(self, logits):
        self.logits = logits

def patch_encoder_for_early_exit(model, early_exit_head, exit_layer=5, entropy_threshold=0.5):
    """
    Patches the ViT layer at `exit_layer` to evaluate the early exit head.
    If the entropy of the predictions is below `entropy_threshold` (i.e. model is confident),
    it raises an EarlyExitException to skip the remaining layers.
    """
    layers = model.vit.layers if hasattr(model.vit, "layers") else model.vit.encoder.layer
    layer_to_patch = layers[exit_layer]
    original_layer_forward = layer_to_patch.forward
    
    def new_layer_forward(*args, **kwargs):
        # Run original forward
        outputs = original_layer_forward(*args, **kwargs)
        
        # outputs is typically (hidden_states, attentions) or just hidden_states
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # EARLY EXIT LOGIC (only active during evaluation)
        if not model.training:
            logits = early_exit_head(hidden_states)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # (B,)
            
            # If all items in batch are confident, exit early
            if (entropy < entropy_threshold).all():
                raise EarlyExitException(logits)
                
        return outputs

    layer_to_patch.forward = new_layer_forward
    model.early_exit_head = early_exit_head

def train_early_exit(model, early_exit_head, train_loader, val_loader, exit_layer=5, epochs=2, device="cuda"):
    """
    Trains ONLY the early exit head, freezing the rest of the model.
    """
    model.to(device)
    early_exit_head.to(device)
    
    model.eval() # Freeze main model
    for param in model.parameters():
        param.requires_grad = False
        
    early_exit_head.train()
    optimizer = torch.optim.AdamW(early_exit_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training early exit head at layer {exit_layer} for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        early_exit_head.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # We need to extract the hidden states at `exit_layer`.
        # The easiest way is to use output_hidden_states=True
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                # Get hidden states without early exiting (since we bypass the patch during training)
                outputs = model.vit(images, output_hidden_states=True)
                # Hidden states index: 0 is embedding, 1 is layer 0, etc.
                # So exit_layer hidden states is at index `exit_layer + 1`
                hidden_states = outputs.hidden_states[exit_layer + 1]
                
            optimizer.zero_grad()
            logits = early_exit_head(hidden_states)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss/(total/images.size(0)), 'acc': correct/total})
            
    print("Early exit head trained!")
    model.to("cpu")
    early_exit_head.to("cpu")
    return early_exit_head
