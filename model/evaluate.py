import os
import sys
import json
import argparse
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script environments
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2


def compute_topk_accuracy(outputs, targets, topk=(1, 5)):
    """Computes Top-K accuracy percentage for given k values."""
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


def load_model(model_path, num_classes=10, device="cpu"):
    """Loads the trained MobileNetV2 model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def plot_confusion_matrix(cm, classes, save_path, title="CIFAR-10 MobileNetV2 Confusion Matrix"):
    """Plots and saves a professional confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    # Normalize matrix for display percentage
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_text = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
            plt.text(
                j, i, cell_text,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    plt.tight_layout()
    plt.ylabel("True Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def evaluate(model, test_loader, classes, device="cpu", is_feature_loader=False):
    """Evaluates the model over the test set, returning predictions and metrics."""
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    top1_correct = 0.0
    top5_correct = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if is_feature_loader:
                outputs = model.classifier(inputs)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            top1, top5 = compute_topk_accuracy(outputs, targets, topk=(1, 5))
            top1_correct += (top1 / 100.0) * targets.size(0)
            top5_correct += (top5 / 100.0) * targets.size(0)
            total_samples += targets.size(0)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    elapsed_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    top1_acc = 100.0 * top1_correct / total_samples
    top5_acc = 100.0 * top5_correct / total_samples

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    return {
        "loss": avg_loss,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "total_samples": total_samples,
        "elapsed_seconds": elapsed_time,
        "predictions": all_preds,
        "targets": all_targets
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 MobileNetV2 Model")
    parser.add_argument("--model-path", type=str, default="model/cifar10_mobilenetv2.pth", help="Path to .pth model")
    parser.add_argument("--classes-path", type=str, default="model/classes.json", help="Path to classes.json")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--raw-images", action="store_true", help="Force evaluating through raw images instead of cache")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("MobileNetV2 CIFAR-10 Model Evaluation")
    print(f"Device: {device}")
    print("=" * 60)

    # 1. Load classes & metadata
    classes = None
    input_size = 64
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if os.path.exists(args.classes_path):
        with open(args.classes_path, "r") as f:
            meta = json.load(f)
            classes = meta.get("classes")
            input_size = meta.get("input_size", 64)
            mean = meta.get("mean", mean)
            std = meta.get("std", std)

    if not classes:
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")

    # 2. Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, num_classes=num_classes, device=device)
    print("Model loaded successfully.")

    # 3. Check for cached test features for fast evaluation
    cache_file = os.path.join(args.data_dir, "feature_cache", f"cifar10_features_size{input_size}.pt")
    use_cache = (not args.raw_images) and os.path.exists(cache_file)

    if use_cache:
        print(f"\nLoading test features from cache: {cache_file}")
        cached_data = torch.load(cache_file)
        test_features = cached_data["test_features"]
        test_targets = cached_data["test_targets"]
        test_loader = DataLoader(
            TensorDataset(test_features, test_targets),
            batch_size=args.batch_size,
            shuffle=False
        )
        is_feature_loader = True
    else:
        print(f"\nLoading raw test images from {args.data_dir}...")
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        is_feature_loader = False

    # 4. Evaluate
    print("Running evaluation on test set (10,000 images)...")
    results = evaluate(model, test_loader, classes, device=device, is_feature_loader=is_feature_loader)

    y_true = results["targets"]
    y_pred = results["predictions"]

    # 5. Compute confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred)
    cm_image_path = os.path.join(args.results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_image_path)

    # Classification report
    clf_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    # 6. Save evaluation metrics JSON
    metrics_summary = {
        "model_path": args.model_path,
        "total_test_samples": results["total_samples"],
        "loss": round(results["loss"], 4),
        "top1_accuracy": round(results["top1_accuracy"], 2),
        "top5_accuracy": round(results["top5_accuracy"], 2),
        "evaluation_time_seconds": round(results["elapsed_seconds"], 2),
        "per_class_accuracy": {
            cls_name: round(clf_report[cls_name]["precision"] * 100.0, 2)
            for cls_name in classes
        },
        "per_class_recall": {
            cls_name: round(clf_report[cls_name]["recall"] * 100.0, 2)
            for cls_name in classes
        },
        "macro_avg_f1": round(clf_report["macro avg"]["f1-score"] * 100.0, 2),
        "weighted_avg_f1": round(clf_report["weighted avg"]["f1-score"] * 100.0, 2)
    }

    metrics_json_path = os.path.join(args.results_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")

    # 7. Print formatted summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test Samples:      {results['total_samples']:,}")
    print(f"Test Loss:         {results['loss']:.4f}")
    print(f"Top-1 Accuracy:    {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy:    {results['top5_accuracy']:.2f}%")
    print(f"Evaluation Time:   {results['elapsed_seconds']:.2f}s")
    print("-" * 60)
    print("Per-Class Performance:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    for cls_name in classes:
        p = clf_report[cls_name]["precision"] * 100.0
        r = clf_report[cls_name]["recall"] * 100.0
        f1 = clf_report[cls_name]["f1-score"] * 100.0
        print(f"{cls_name:<12} {p:6.2f}%       {r:6.2f}%       {f1:6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
