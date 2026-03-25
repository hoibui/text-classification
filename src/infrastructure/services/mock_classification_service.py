import asyncio
import random
from typing import List, Dict
from datetime import datetime

from domain.entities.text_classification import TextClassification, ClassificationResult
from domain.services.classification_service import ClassificationService


class MockClassificationService(ClassificationService):
    """Mock classification service for development and testing."""

    def __init__(self):
        self.mock_labels = ["CATEGORY_A", "CATEGORY_B", "CATEGORY_C"]
        self.model_version = "mock-1.0.0"

    async def classify_text(self, text_classification: TextClassification) -> ClassificationResult:
        """Classify a single text and return the result."""
        # Simulate some processing time
        await asyncio.sleep(0.01)

        # Generate mock prediction
        predicted_label = random.choice(self.mock_labels)
        confidence = random.uniform(0.7, 0.95)

        # Generate mock probabilities
        probabilities = {}
        remaining_prob = 1.0 - confidence
        for label in self.mock_labels:
            if label == predicted_label:
                probabilities[label] = confidence
            else:
                prob = random.uniform(0, remaining_prob / (len(self.mock_labels) - 1))
                probabilities[label] = prob
                remaining_prob -= prob

        # Normalize probabilities
        total_prob = sum(probabilities.values())
        probabilities = {k: v / total_prob for k, v in probabilities.items()}

        return ClassificationResult(
            text=text_classification.text,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            model_version=self.model_version,
            processing_time=0.01,
            request_id=text_classification.request_id,
            timestamp=datetime.utcnow()
        )

    async def classify_batch(self, text_classifications: List[TextClassification]) -> List[ClassificationResult]:
        """Classify multiple texts and return the results."""
        results = []
        for text_classification in text_classifications:
            result = await self.classify_text(text_classification)
            results.append(result)
        return results

    async def get_model_info(self) -> Dict[str, any]:
        """Get information about the currently loaded model."""
        return {
            "model_name": "mock_classifier",
            "model_version": self.model_version,
            "model_type": "mock",
            "status": "ready",
            "loaded_at": datetime.utcnow().isoformat(),
            "supported_labels": self.mock_labels
        }

    async def is_model_ready(self) -> bool:
        """Check if the classification model is ready for inference."""
        return True

    async def reload_model(self, model_path: str = None) -> bool:
        """Reload the classification model. Returns True if successful."""
        # Mock reload - always successful
        return True