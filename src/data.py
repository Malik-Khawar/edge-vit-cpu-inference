import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_imagenette_data(batch_size=32, data_dir="../data", img_size=224, use_synthetic=False):
    """
    Loads benchmark dataset.
    - If `use_synthetic=False` (default): Downloads/loads real CIFAR-10 dataset (resized to 224x224).
    - If `use_synthetic=True` or if download fails/throttles: Uses fast synthetic (3, 224, 224) tensors for zero-latency CPU profiling.
    """
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not use_synthetic:
        try:
            print("Downloading/Loading real CIFAR-10 dataset...")
            train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
            val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
            val_dataset = Subset(val_dataset, list(range(min(500, len(val_dataset)))))
            print(f"Loaded real dataset successfully ({len(train_dataset)} train samples).")
        except Exception as e:
            print(f"Notice: Real dataset download failed or throttled ({e}).")
            use_synthetic = True

    if use_synthetic:
        print("Using fast synthetic benchmark image dataset (3x224x224 tensors) for zero-latency CPU profiling...")
        import torch
        from torch.utils.data import TensorDataset
        
        # Synthetic images matching standard ViT input shape (C, H, W) = (3, 224, 224)
        x_train = torch.randn(200, 3, img_size, img_size)
        y_train = torch.randint(0, 1000, (200,))
        x_val = torch.randn(100, 3, img_size, img_size)
        y_val = torch.randint(0, 1000, (100,))
        
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
