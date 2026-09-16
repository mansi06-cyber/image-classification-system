import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import PredictResponse, ClassesResponse, ModelInfoResponse
from backend.classifier import get_classifier, ImageClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the MobileNetV2 model once during application startup.
    Ensures model weights are in memory before handling incoming requests.
    """
    print("[Backend] Initializing and loading MobileNetV2 model into memory...")
    classifier = get_classifier()
    app.state.classifier = classifier
    print(f"[Backend] Startup complete. Model ready for inference on {classifier.device}.")
    yield
    print("[Backend] Shutting down application...")


app = FastAPI(
    title="Image Classification System API",
    description="FastAPI backend serving ImageNet-pretrained MobileNetV2 fine-tuned on CIFAR-10",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint verifying API availability."""
    return {
        "status": "online",
        "message": "CIFAR-10 Image Classification System API is running",
        "endpoints": {
            "predict": "POST /api/predict",
            "classes": "GET /api/classes",
            "model_info": "GET /api/model/info",
            "documentation": "/docs"
        }
    }


@app.get("/api/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/classes", response_model=ClassesResponse, tags=["Metadata"])
async def get_classes():
    """
    Returns the list of all 10 CIFAR-10 target classes supported by the model.
    """
    classifier: ImageClassifier = get_classifier()
    return ClassesResponse(
        classes=classifier.classes,
        total_classes=len(classifier.classes)
    )


@app.get("/api/model/info", response_model=ModelInfoResponse, tags=["Metadata"])
async def get_model_info():
    """
    Returns metadata about the model architecture, training performance,
    dataset, input requirements, and file specifications.
    """
    classifier: ImageClassifier = get_classifier()
    return classifier.get_model_info()


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_image(file: UploadFile = File(None)):
    """
    Upload an image (JPEG or PNG) to classify it using MobileNetV2.
    Returns the predicted class, confidence score, Top-5 ranking, and inference latency.
    """
    # 1. Check for missing upload
    if file is None or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please upload a JPEG or PNG image file using form key 'file'."
        )

    # 2. Read file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # 3. Check for empty file
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty. Please upload a valid image."
        )

    # 4. Check for oversized file
    if len(file_bytes) > 10 * 1024 * 1024:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded image is too large ({size_mb:.2f} MB). Maximum allowed size is 10 MB."
        )

    # 5. Run inference with validation
    classifier: ImageClassifier = get_classifier()
    try:
        prediction_result = classifier.predict(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type or ""
        )
        return prediction_result
    except ValueError as val_err:
        err_msg = str(val_err)
        # Match error type to suitable HTTP status
        if "Unsupported file extension" in err_msg or "Unsupported media type" in err_msg or "Unsupported image format" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=err_msg
            )
        elif "exceeds maximum allowed limit" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=err_msg
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err_msg
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )
