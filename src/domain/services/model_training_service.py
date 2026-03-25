from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..entities.model import Model, ModelMetrics


class ModelTrainingService(ABC):
    """Domain service interface for model training."""

    @abstractmethod
    async def train_model(
        self,
        training_data_path: str,
        model_config: Dict[str, Any],
        model_name: str,
        version: str
    ) -> Model:
        """Train a new model and return the trained model entity."""
        pass

    @abstractmethod
    async def evaluate_model(self, model: Model, test_data_path: str) -> ModelMetrics:
        """Evaluate a model and return performance metrics."""
        pass

    @abstractmethod
    async def compare_models(self, model_1: Model, model_2: Model) -> Dict[str, Any]:
        """Compare two models and return comparison metrics."""
        pass

    @abstractmethod
    async def get_training_progress(self, training_id: str) -> Dict[str, Any]:
        """Get the progress of an ongoing training process."""
        pass

    @abstractmethod
    async def cancel_training(self, training_id: str) -> bool:
        """Cancel an ongoing training process."""
        pass

    @abstractmethod
    async def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """Validate a model training configuration."""
        pass