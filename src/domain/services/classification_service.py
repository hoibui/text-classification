from abc import ABC, abstractmethod
from typing import List, Dict
from ..entities.text_classification import TextClassification, ClassificationResult


class ClassificationService(ABC):
    """Domain service interface for text classification."""

    @abstractmethod
    async def classify_text(self, text_classification: TextClassification) -> ClassificationResult:
        """Classify a single text and return the result."""
        pass

    @abstractmethod
    async def classify_batch(self, text_classifications: List[TextClassification]) -> List[ClassificationResult]:
        """Classify multiple texts and return the results."""
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, any]:
        """Get information about the currently loaded model."""
        pass

    @abstractmethod
    async def is_model_ready(self) -> bool:
        """Check if the classification model is ready for inference."""
        pass

    @abstractmethod
    async def reload_model(self, model_path: str = None) -> bool:
        """Reload the classification model. Returns True if successful."""
        pass