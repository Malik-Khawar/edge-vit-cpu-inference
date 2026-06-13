import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_imagenette_data(batch_size=32, data_dir="../data", img_size=224):
    """
    Downloads CIFAR-10 (as a fast proxy for Imagenette) and resizes to 224x224.
    Since we are benchmarking Latency, RAM, and FLOPs, the specific dataset content 
    is less important than the image dimensions (3, 224, 224).
    """
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Downloading CIFAR-10 for benchmarking...")
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Use full training dataset for GPU training, and subset validation dataset
    # to 500 images to keep the CPU feature similarity evaluation fast
    val_dataset = Subset(val_dataset, list(range(500)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
