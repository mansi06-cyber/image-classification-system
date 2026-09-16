import os
import io
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

from backend.schemas import TopPrediction, PredictResponse, ModelInfoResponse

# Maximum allowed upload size (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}


class ImageClassifier:
    """
    Handles MobileNetV2 model loading, image validation, preprocessing,
    and CPU inference for CIFAR-10 classification.
    """

    def __init__(
        self,
        model_path: str = "model/cifar10_mobilenetv2.pth",
        classes_path: str = "model/classes.json"
    ):
        self.device = torch.device("cpu")
        self.model_path = model_path
        self.classes_path = classes_path

        # 1. Load metadata and class names
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
        self.input_size = 64
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.metadata = {}

        self._load_metadata()

        # 2. Build preprocessing pipeline matching training
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # 3. Load model weights
        self.model = self._load_model()
        print(f"[Classifier] MobileNetV2 loaded successfully on {self.device} with {len(self.classes)} classes.")

    def _load_metadata(self) -> None:
        """Loads classes and training configurations from classes.json if present."""
        if os.path.exists(self.classes_path):
            try:
                with open(self.classes_path, "r") as f:
                    self.metadata = json.load(f)
                    self.classes = self.metadata.get("classes", self.classes)
                    self.input_size = self.metadata.get("input_size", self.input_size)
                    self.mean = self.metadata.get("mean", self.mean)
                    self.std = self.metadata.get("std", self.std)
            except Exception as e:
                print(f"[Classifier] Warning loading classes.json: {e}")

    def _load_model(self) -> nn.Module:
        """Initializes MobileNetV2 architecture and loads trained weights."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        model = mobilenet_v2()
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(self.classes))

        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def validate_and_preprocess(self, file_bytes: bytes, filename: str = "", content_type: str = "") -> torch.Tensor:
        """
        Validates file format, size, and image integrity, then converts to a model-ready tensor.
        Raises ValueError with descriptive messages for invalid inputs.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError("No file content uploaded. Please upload a non-empty image file.")

        # 1. Size check
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            size_mb = len(file_bytes) / (1024 * 1024)
            raise ValueError(f"Image size ({size_mb:.2f} MB) exceeds maximum allowed limit of 10 MB.")

        # 2. Extension check
        if filename:
            ext = os.path.splitext(filename.lower())[1]
            if ext and ext not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Unsupported file extension '{ext}'. Only JPEG and PNG (.jpg, .jpeg, .png) images are supported.")

        # 3. Content type check
        if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
            # If extension is valid but MIME type is octet-stream, allow checking image headers
            if content_type != "application/octet-stream":
                raise ValueError(f"Unsupported media type '{content_type}'. Expected image/jpeg or image/png.")

        # 4. Image decoding & corruption check
        try:
            image = Image.open(io.BytesIO(file_bytes))
            # Verify image format
            if image.format not in ("JPEG", "PNG"):
                raise ValueError(f"Unsupported image format '{image.format}'. Only JPEG and PNG are supported.")
            # Convert to RGB (handles RGBA, grayscale, or palette images)
            image = image.convert("RGB")
            # Force load image data to catch truncation or corruption
            image.load()
        except UnidentifiedImageError:
            raise ValueError("The uploaded file is not a valid or readable image.")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Corrupted or unreadable image file: {str(e)}")

        # 5. Apply transforms
        tensor = self.transform(image).unsqueeze(0)  # Shape: (1, 3, H, W)
        return tensor

    def predict(self, file_bytes: bytes, filename: str = "", content_type: str = "") -> PredictResponse:
        """
        Runs CPU inference on image bytes and returns top prediction, top-5 ranking, and latency.
        """
        # Validate and preprocess
        tensor = self.validate_and_preprocess(file_bytes, filename=filename, content_type=content_type)
        tensor = tensor.to(self.device)

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Extract Top-5 predictions
        k = min(5, len(self.classes))
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top_5_list: List[TopPrediction] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cls_name = self.classes[idx]
            conf = float(prob)
            top_5_list.append(
                TopPrediction(
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    percentage=round(conf * 100.0, 2)
                )
            )

        top_prediction = top_5_list[0]

        return PredictResponse(
            predicted_class=top_prediction.class_name,
            confidence_score=top_prediction.confidence,
            confidence_percentage=top_prediction.percentage,
            top_5=top_5_list,
            inference_time_ms=inference_time_ms
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Returns metadata, accuracy metrics, and architecture info."""
        size_mb = 0.0
        mod_date_str = None

        if os.path.exists(self.model_path):
            stat = os.stat(self.model_path)
            size_mb = round(stat.st_size / (1024 * 1024), 2)
            mod_date_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        top1 = f"{self.metadata.get('best_test_top1_accuracy', 76.20):.2f}%"
        top5 = f"{self.metadata.get('best_test_top5_accuracy', 98.71):.2f}%"

        return ModelInfoResponse(
            architecture="MobileNetV2",
            dataset="CIFAR-10",
            number_of_classes=len(self.classes),
            top_1_accuracy=top1,
            top_5_accuracy=top5,
            model_file=self.model_path,
            model_size_mb=size_mb,
            input_size=f"{self.input_size}x{self.input_size}",
            device="CPU",
            training_date=mod_date_str,
            status="loaded"
        )


# Global singleton instance
_classifier_instance: ImageClassifier = None


def get_classifier() -> ImageClassifier:
    """Returns the singleton classifier instance, loading it if not already loaded."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ImageClassifier()
    return _classifier_instance
