from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime


@dataclass
class ClassificationRequestDTO:
    """Data Transfer Object for classification requests."""
    text: str
    return_confidence: bool = False
    request_id: Optional[str] = None


@dataclass
class BatchClassificationRequestDTO:
    """Data Transfer Object for batch classification requests."""
    texts: List[str]
    return_confidence: bool = False
    batch_id: Optional[str] = None


@dataclass
class ClassificationResponseDTO:
    """Data Transfer Object for classification responses."""
    text: str
    predicted_label: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    processing_time: Optional[float] = None
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    @classmethod
    def from_domain(cls, domain_result) -> 'ClassificationResponseDTO':
        """Create DTO from domain entity."""
        from domain.entities.text_classification import ClassificationResult

        if isinstance(domain_result, ClassificationResult):
            return cls(
                text=domain_result.text,
                predicted_label=domain_result.predicted_label,
                confidence=domain_result.confidence,
                probabilities=domain_result.probabilities,
                model_version=domain_result.model_version,
                processing_time=domain_result.processing_time,
                request_id=domain_result.request_id,
                timestamp=domain_result.timestamp
            )
        raise ValueError("Invalid domain result type")