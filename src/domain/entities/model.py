from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class ModelStatus(Enum):
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetrics:
    """Domain entity representing model performance metrics."""
    accuracy: float
    f1_score: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Validate metrics are within valid ranges
        for metric_name, value in [("accuracy", self.accuracy), ("f1_score", self.f1_score)]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{metric_name} must be between 0.0 and 1.0, got {value}")


@dataclass
class Model:
    """Domain entity representing a machine learning model."""
    id: Optional[str]
    name: str
    version: str
    model_type: str  # e.g., "logistic_regression", "random_forest", "transformer"
    status: ModelStatus
    metrics: Optional[ModelMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    file_path: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

    def update_status(self, new_status: ModelStatus) -> None:
        """Update model status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def is_ready_for_inference(self) -> bool:
        """Check if model is ready for making predictions."""
        return self.status in [ModelStatus.READY, ModelStatus.DEPLOYED]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of model performance."""
        if not self.metrics:
            return {}

        return {
            "accuracy": self.metrics.accuracy,
            "f1_score": self.metrics.f1_score,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            **self.metrics.additional_metrics
        }