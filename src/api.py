from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import logging
import os
import time
from datetime import datetime
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(
    title="Charge Type Classification API",
    description="Professional API for classifying utility charge types",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions made')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Time spent on predictions')
ERROR_COUNTER = Counter('errors_total', 'Total number of errors', ['error_type'])

class PredictionRequest(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1)
    return_confidence: bool = Field(True, description="Whether to return confidence scores")

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to classify", min_items=1)
    return_confidence: bool = Field(True, description="Whether to return confidence scores")

class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: Optional[float] = None
    processing_time: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time: float

class ModelInfo(BaseModel):
    model_key: str
    score: float
    classes: List[str]
    model_size: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    uptime: float

class ModelManager:
    def __init__(self):
        self.model_data = None
        self.model_path = None
        self.load_time = None
        self.start_time = time.time()

    def load_model(self, model_path: str = "models/best_charge_classifier.pkl"):
        try:
            if not os.path.exists(model_path):
                available_models = [f for f in os.listdir("models") if f.endswith(".pkl")]
                if available_models:
                    model_path = os.path.join("models", available_models[-1])  # Use latest
                    logger.info(f"Using latest available model: {model_path}")
                else:
                    raise FileNotFoundError("No model files found in models directory")

            start_time = time.time()
            self.model_data = joblib.load(model_path)
            self.model_path = model_path
            self.load_time = time.time() - start_time
            logger.info(f"Model loaded successfully from {model_path} in {self.load_time:.3f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            ERROR_COUNTER.labels(error_type="model_load").inc()
            return False

    def predict(self, text: str, return_confidence: bool = True) -> Dict[str, Any]:
        if not self.model_data:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()

        try:
            model = self.model_data['model']
            vectorizer = self.model_data['vectorizer']
            label_encoder = self.model_data['label_encoder']

            text_cleaned = str(text).lower().strip()
            X_vec = vectorizer.transform([text_cleaned])
            prediction = model.predict(X_vec)[0]

            predicted_label = label_encoder.inverse_transform([prediction])[0]

            result = {
                "predicted_label": predicted_label,
                "processing_time": time.time() - start_time
            }

            if return_confidence:
                probabilities = model.predict_proba(X_vec)[0]
                confidence = float(np.max(probabilities))
                result["confidence"] = confidence

            PREDICTION_COUNTER.inc()
            PREDICTION_DURATION.observe(result["processing_time"])

            return result

        except Exception as e:
            ERROR_COUNTER.labels(error_type="prediction").inc()
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        if not self.model_data:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return {
            "model_key": self.model_data.get('model_key', 'unknown'),
            "score": self.model_data.get('score', 0.0),
            "classes": self.model_data['label_encoder'].classes_.tolist(),
            "model_size": f"{os.path.getsize(self.model_path) / (1024*1024):.2f} MB" if self.model_path else "unknown"
        }

    def get_uptime(self) -> float:
        return time.time() - self.start_time

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Charge Type Classification API...")
    if not model_manager.load_model():
        logger.warning("Failed to load model on startup. API will be in degraded state.")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager.model_data else "degraded",
        timestamp=datetime.now(),
        model_loaded=model_manager.model_data is not None,
        uptime=model_manager.get_uptime()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    result = model_manager.predict(request.text, request.return_confidence)
    return PredictionResponse(**result)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    start_time = time.time()
    predictions = []

    for text in request.texts:
        try:
            result = model_manager.predict(text, request.return_confidence)
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Failed to predict for text: {text[:50]}... Error: {e}")
            ERROR_COUNTER.labels(error_type="batch_prediction").inc()
            predictions.append(PredictionResponse(
                predicted_label="ERROR",
                confidence=0.0 if request.return_confidence else None,
                processing_time=0.0
            ))

    total_time = time.time() - start_time

    return BatchPredictionResponse(
        predictions=predictions,
        total_processing_time=total_time
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    return ModelInfo(**model_manager.get_model_info())

@app.post("/model/reload")
async def reload_model(model_path: Optional[str] = None):
    if model_path is None:
        model_path = "models/best_charge_classifier.pkl"

    success = model_manager.load_model(model_path)
    if success:
        return {"message": f"Model reloaded successfully from {model_path}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
async def root():
    return {
        "message": "Charge Type Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)