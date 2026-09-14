# Image Classification System

An end-to-end Image Classification System using MobileNetV2 trained on CIFAR-10, served via a FastAPI backend with an interactive web frontend.

## Tech Stack
- **Deep Learning**: Python, PyTorch, torchvision (MobileNetV2, CIFAR-10)
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML5, CSS3, JavaScript (Fetch API, LocalStorage)
- **Metrics & Evaluation**: scikit-learn, Matplotlib, Seaborn

## Project Structure
```text
image-classification-system/
├── backend/
│   ├── __init__.py
│   ├── classifier.py
│   ├── main.py
│   └── schemas.py
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── model/
│   ├── __init__.py
│   ├── evaluate.py
│   └── train.py
├── notebooks/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── README.md
└── requirements.txt
```

## Planned API Endpoints
- `POST /api/predict`: Upload an image (JPEG/PNG) and get Top-1 & Top-5 predictions with confidence scores.
- `GET /api/classes`: Returns list of CIFAR-10 class labels.
- `GET /api/model/info`: Returns model metadata, architecture details, and status.

## Key Requirements
- Image upload with JPEG/PNG validation and client-side preview.
- Error handling for corrupted, oversized, or unsupported files.
- Top-1 and Top-5 accuracy evaluation and confusion matrix plot saved in `results/`.
- Fast inference latency under 2 seconds per image.
- Prediction history persisted in browser `localStorage`.
