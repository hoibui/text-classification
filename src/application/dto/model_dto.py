from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class ModelMetricsDTO:
    """Data Transfer Object for model metrics."""
    accuracy: float
    f1_score: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelDTO:
    """Data Transfer Object for model information."""
    id: Optional[str]
    name: str
    version: str
    model_type: str
    status: str
    metrics: Optional[ModelMetricsDTO] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    file_path: Optional[str] = None

    @classmethod
    def from_domain(cls, domain_model) -> 'ModelDTO':
        """Create DTO from domain entity."""
        from domain.entities.model import Model

        if isinstance(domain_model, Model):
            metrics_dto = None
            if domain_model.metrics:
                metrics_dto = ModelMetricsDTO(
                    accuracy=domain_model.metrics.accuracy,
                    f1_score=domain_model.metrics.f1_score,
                    precision=domain_model.metrics.precision,
                    recall=domain_model.metrics.recall,
                    additional_metrics=domain_model.metrics.additional_metrics
                )

            return cls(
                id=domain_model.id,
                name=domain_model.name,
                version=domain_model.version,
                model_type=domain_model.model_type,
                status=domain_model.status.value,
                metrics=metrics_dto,
                metadata=domain_model.metadata,
                created_at=domain_model.created_at,
                updated_at=domain_model.updated_at,
                file_path=domain_model.file_path
            )
        raise ValueError("Invalid domain model type")


@dataclass
class ModelTrainingRequestDTO:
    """Data Transfer Object for model training requests."""
    name: str
    version: str
    model_type: str
    training_data_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None