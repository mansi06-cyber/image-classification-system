from typing import List, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_name: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Confidence probability between 0 and 1")
    percentage: float = Field(..., description="Confidence expressed as percentage 0-100%")


class PredictResponse(BaseModel):
    predicted_class: str = Field(..., description="Top predicted class name")
    confidence_score: float = Field(..., description="Confidence probability of top prediction (0-1)")
    confidence_percentage: float = Field(..., description="Confidence percentage of top prediction (0-100%)")
    top_5: List[TopPrediction] = Field(..., description="Top 5 predictions ranked by confidence")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")


class ClassesResponse(BaseModel):
    classes: List[str] = Field(..., description="List of all supported CIFAR-10 classes")
    total_classes: int = Field(..., description="Total number of classes")


class ModelInfoResponse(BaseModel):
    architecture: str = Field(..., description="Model architecture name")
    dataset: str = Field(..., description="Dataset model was trained on")
    number_of_classes: int = Field(..., description="Number of output classes")
    top_1_accuracy: str = Field(..., description="Test Top-1 accuracy percentage")
    top_5_accuracy: str = Field(..., description="Test Top-5 accuracy percentage")
    model_file: str = Field(..., description="Relative path to model weights file")
    model_size_mb: float = Field(..., description="Size of model file in megabytes")
    input_size: str = Field(..., description="Input resolution expected by model")
    device: str = Field(..., description="Compute device used for inference")
    training_date: Optional[str] = Field(None, description="Date/timestamp model was trained or modified")
    status: str = Field(..., description="Current status of model service")
