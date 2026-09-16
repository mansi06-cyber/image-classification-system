import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def compute_topk_accuracy(outputs, targets, topk=(1, 5)):
    """Computes Top-K accuracy for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size * 100.0).item())
        return res


def extract_features(feature_model, dataloader, device, desc="Extracting features"):
    """
    Extracts 1280-dim feature vectors from the frozen MobileNetV2 backbone.
    Caches features in memory to accelerate multi-epoch classifier training by 5-10x on CPU.
    """
    feature_model.eval()
    all_features = []
    all_targets = []
    total_samples = len(dataloader.dataset)
    processed = 0
    start_time = time.time()

    print(f"\n{desc} ({total_samples:,} images)...")
    sys.stdout.flush()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)

            # Pass through MobileNetV2 features + pooling + flatten
            feats = feature_model.features(images)
            feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            feats = torch.flatten(feats, 1)

            all_features.append(feats.cpu())
            all_targets.append(targets)
            processed += images.size(0)

            if (batch_idx + 1) % 50 == 0 or processed == total_samples:
                elapsed = time.time() - start_time
                fps = processed / elapsed if elapsed > 0 else 0
                pct = 100.0 * processed / total_samples
                print(f"  [{processed:5d}/{total_samples:5d}] ({pct:5.1f}%) - {fps:5.1f} img/s")
                sys.stdout.flush()

    features_tensor = torch.cat(all_features, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    return features_tensor, targets_tensor


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and extraction")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--img-size", type=int, default=64, help="Input image size (default 64x64 for optimal CPU balance)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to CIFAR-10 data")
    parser.add_argument("--save-path", type=str, default="model/cifar10_mobilenetv2.pth", help="Path to save best model")
    parser.add_argument("--cache-features", action="store_true", default=True, help="Cache extracted features to disk")
    args = parser.parse_args()

    # Configure hardware threads for CPU optimization
    if not torch.cuda.is_available():
        torch.set_num_threads(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"MobileNetV2 CIFAR-10 Training Pipeline")
    print(f"Device: {device} | Cores/Threads: {torch.get_num_threads()} | Input Size: {args.img_size}x{args.img_size}")
    print("=" * 60)

    # 1. Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 2. Load full CIFAR-10 datasets
    print("\nLoading CIFAR-10 datasets...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform
    )

    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Test samples:  {len(test_dataset):,}")
    print(f"Classes ({num_classes}): {classes}")

    # 3. Load pretrained MobileNetV2
    print("\nLoading ImageNet-pretrained MobileNetV2...")
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head with CIFAR-10 classes (10 outputs)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    print("Model Classifier:")
    print(model.classifier)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")

    # 4. Feature Extraction & Caching
    # Since feature extractor is frozen, extracting features once saves hours of redundant CPU passes.
    cache_dir = os.path.join(args.data_dir, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"cifar10_features_size{args.img_size}.pt")

    if args.cache_features and os.path.exists(cache_file):
        print(f"\nLoading cached features from: {cache_file}")
        cached_data = torch.load(cache_file)
        train_features = cached_data["train_features"]
        train_targets = cached_data["train_targets"]
        test_features = cached_data["test_features"]
        test_targets = cached_data["test_targets"]
        print(f"Loaded {len(train_features):,} train and {len(test_features):,} test feature vectors.")
    else:
        extract_loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        extract_loader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        train_features, train_targets = extract_features(
            model, extract_loader_train, device, desc="Extracting train features"
        )
        test_features, test_targets = extract_features(
            model, extract_loader_test, device, desc="Extracting test features"
        )

        if args.cache_features:
            print(f"Saving extracted features to cache: {cache_file}")
            torch.save({
                "train_features": train_features,
                "train_targets": train_targets,
                "test_features": test_features,
                "test_targets": test_targets
            }, cache_file)

    # 5. Fast Feature DataLoaders
    train_feat_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_feat_loader = DataLoader(
        TensorDataset(test_features, test_targets),
        batch_size=args.batch_size,
        shuffle=False
    )

    # 6. Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    # 7. Training and Evaluation Loop
    print("\n" + "=" * 60)
    print(f"Starting Training for {args.epochs} Epochs")
    print("=" * 60)

    best_top1_acc = 0.0
    best_top5_acc = 0.0
    best_epoch = 0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # --- Training ---
        model.classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for feats, targets in train_feat_loader:
            feats, targets = feats.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model.classifier(feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * feats.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- Evaluation on Test Set ---
        model.classifier.eval()
        test_loss = 0.0
        top1_correct = 0.0
        top5_correct = 0.0
        test_total = 0

        with torch.no_grad():
            for feats, targets in test_feat_loader:
                feats, targets = feats.to(device), targets.to(device)
                outputs = model.classifier(feats)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * feats.size(0)
                top1, top5 = compute_topk_accuracy(outputs, targets, topk=(1, 5))
                top1_correct += (top1 / 100.0) * targets.size(0)
                top5_correct += (top5 / 100.0) * targets.size(0)
                test_total += targets.size(0)

        epoch_test_loss = test_loss / test_total
        epoch_top1 = 100.0 * top1_correct / test_total
        epoch_top5 = 100.0 * top5_correct / test_total
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch}/{args.epochs}] ({epoch_time:5.2f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:6.2f}% | "
            f"Test Loss: {epoch_test_loss:.4f} | Test Top-1: {epoch_top1:6.2f}% | Test Top-5: {epoch_top5:6.2f}%"
        )
        sys.stdout.flush()

        # Save best model based on test Top-1 accuracy
        if epoch_top1 > best_top1_acc:
            best_top1_acc = epoch_top1
            best_top5_acc = epoch_top5
            best_epoch = epoch

            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  --> Saved new best model to '{args.save_path}' (Top-1: {best_top1_acc:.2f}%, Top-5: {best_top5_acc:.2f}%)")
            sys.stdout.flush()

    total_training_time = time.time() - training_start

    # 8. Save Model Metadata and Class Mapping
    meta_path = os.path.join(os.path.dirname(args.save_path), "classes.json")
    model_metadata = {
        "model_name": "MobileNetV2",
        "num_classes": num_classes,
        "classes": classes,
        "input_size": args.img_size,
        "mean": mean,
        "std": std,
        "best_epoch": best_epoch,
        "best_test_top1_accuracy": round(best_top1_acc, 2),
        "best_test_top5_accuracy": round(best_top5_acc, 2),
        "total_training_time_seconds": round(total_training_time, 2)
    }
    with open(meta_path, "w") as f:
        json.dump(model_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Pipeline Completed Successfully!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Test Top-1 Accuracy: {best_top1_acc:.2f}%")
    print(f"Best Test Top-5 Accuracy: {best_top5_acc:.2f}%")
    print(f"Total Time: {total_training_time:.2f}s")
    print(f"Model saved at: {args.save_path}")
    print(f"Metadata saved at: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
