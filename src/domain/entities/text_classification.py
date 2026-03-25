from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class TextClassification:
    """Domain entity representing a text classification request."""
    text: str
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ClassificationResult:
    """Domain entity representing the result of text classification."""
    text: str
    predicted_label: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    processing_time: Optional[float] = None
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if the prediction confidence is above threshold."""
        return self.confidence >= threshold

    def get_metadata(self) -> Dict[str, Any]:
        """Get classification metadata."""
        return {
            "model_version": self.model_version,
            "processing_time": self.processing_time,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence
        }