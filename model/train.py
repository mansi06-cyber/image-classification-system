import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# 1. Image preprocessing appropriate for MobileNetV2
# MobileNetV2 expects 224x224 input images normalized with ImageNet mean and standard deviation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2. Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

classes = train_dataset.classes
num_classes = len(classes)

print(f"Training images: {len(train_dataset)}")
print(f"Testing images: {len(test_dataset)}")
print(f"Classes ({num_classes}): {classes}")

# 3. Load MobileNetV2 with ImageNet pretrained weights
print("\nLoading pretrained MobileNetV2...")
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)

# 4. Replace final classifier for CIFAR-10 (10 classes)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# 5. Inspect the updated classifier
print("\nModel final classifier:")
print(model.classifier)
print(f"\nNumber of output classes: {model.classifier[1].out_features}")
