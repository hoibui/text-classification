from typing import Dict, Any
import uuid
from datetime import datetime

from domain.entities.model import Model, ModelStatus, ModelMetrics
from domain.services.model_training_service import ModelTrainingService
from domain.repositories.model_repository import ModelRepository
from application.dto.model_dto import ModelTrainingRequestDTO, ModelDTO


class TrainModelUseCase:
    """Use case for model training operations."""

    def __init__(
        self,
        model_training_service: ModelTrainingService,
        model_repository: ModelRepository
    ):
        self._training_service = model_training_service
        self._model_repository = model_repository

    async def execute(self, request: ModelTrainingRequestDTO) -> ModelDTO:
        """Execute model training."""
        # Validate configuration
        is_valid = await self._training_service.validate_model_config(request.config)
        if not is_valid:
            raise ValueError("Invalid model configuration")

        # Check if model with same name/version already exists
        existing_model = await self._model_repository.find_by_name_and_version(
            request.name, request.version
        )
        if existing_model:
            raise ValueError(f"Model {request.name} version {request.version} already exists")

        # Train the model
        trained_model = await self._training_service.train_model(
            training_data_path=request.training_data_path,
            model_config=request.config,
            model_name=request.name,
            version=request.version
        )

        # Save the trained model
        saved_model = await self._model_repository.save(trained_model)

        return ModelDTO.from_domain(saved_model)

    async def get_training_progress(self, training_id: str) -> Dict[str, Any]:
        """Get training progress for a specific training job."""
        return await self._training_service.get_training_progress(training_id)

    async def cancel_training(self, training_id: str) -> bool:
        """Cancel an ongoing training job."""
        return await self._training_service.cancel_training(training_id)

    async def evaluate_model(self, model_id: str, test_data_path: str) -> Dict[str, Any]:
        """Evaluate a trained model."""
        model = await self._model_repository.find_by_id(model_id)
        if not model:
            raise ValueError(f"Model with ID {model_id} not found")

        metrics = await self._training_service.evaluate_model(model, test_data_path)

        # Update model with evaluation metrics
        model.metrics = metrics
        model.update_status(ModelStatus.READY)
        await self._model_repository.update(model)

        return {
            "accuracy": metrics.accuracy,
            "f1_score": metrics.f1_score,
            "precision": metrics.precision,
            "recall": metrics.recall,
            **metrics.additional_metrics
        }