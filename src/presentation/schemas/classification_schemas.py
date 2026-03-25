from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ClassificationRequest(BaseModel):
    """Request schema for single text classification."""
    text: str = Field(..., description="Text to classify", min_length=1, max_length=10000)
    return_confidence: bool = Field(default=False, description="Whether to return confidence scores")
    request_id: Optional[str] = Field(default=None, description="Optional request ID for tracking")


class BatchClassificationRequest(BaseModel):
    """Request schema for batch text classification."""
    texts: List[str] = Field(..., description="List of texts to classify", min_items=1, max_items=100)
    return_confidence: bool = Field(default=False, description="Whether to return confidence scores")
    batch_id: Optional[str] = Field(default=None, description="Optional batch ID for tracking")


class ClassificationResponse(BaseModel):
    """Response schema for text classification."""
    text: str
    predicted_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    processing_time: Optional[float] = None
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    timestamp: datetime
    version: str
    model_loaded: bool
    database_connected: bool


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_name: str
    model_version: str
    model_type: str
    status: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    loaded_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}