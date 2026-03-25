from typing import List, Optional, Dict, Any

from domain.entities.model import ModelStatus
from domain.repositories.model_repository import ModelRepository
from domain.services.classification_service import ClassificationService
from application.dto.model_dto import ModelDTO


class GetModelInfoUseCase:
    """Use case for retrieving model information."""

    def __init__(
        self,
        model_repository: ModelRepository,
        classification_service: ClassificationService
    ):
        self._model_repository = model_repository
        self._classification_service = classification_service

    async def get_active_model(self, model_name: str) -> Optional[ModelDTO]:
        """Get the currently active model by name."""
        model = await self._model_repository.find_active_by_name(model_name)
        if not model:
            return None
        return ModelDTO.from_domain(model)

    async def get_model_by_id(self, model_id: str) -> Optional[ModelDTO]:
        """Get a model by its ID."""
        model = await self._model_repository.find_by_id(model_id)
        if not model:
            return None
        return ModelDTO.from_domain(model)

    async def list_models(self, limit: Optional[int] = None) -> List[ModelDTO]:
        """List all models."""
        models = await self._model_repository.list_all(limit)
        return [ModelDTO.from_domain(model) for model in models]

    async def list_models_by_status(self, status: str) -> List[ModelDTO]:
        """List models by status."""
        try:
            model_status = ModelStatus(status)
        except ValueError:
            raise ValueError(f"Invalid status: {status}")

        models = await self._model_repository.find_by_status(model_status)
        return [ModelDTO.from_domain(model) for model in models]

    async def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model in the classification service."""
        return await self._classification_service.get_model_info()

    async def set_active_model(self, model_id: str, model_name: str) -> bool:
        """Set a model as the active one."""
        # Verify model exists and is ready
        model = await self._model_repository.find_by_id(model_id)
        if not model:
            raise ValueError(f"Model with ID {model_id} not found")

        if not model.is_ready_for_inference():
            raise ValueError(f"Model {model_id} is not ready for inference (status: {model.status.value})")

        # Set as active in repository
        success = await self._model_repository.set_active(model_id, model_name)
        if not success:
            return False

        # Reload model in classification service
        reload_success = await self._classification_service.reload_model(model.file_path)
        if not reload_success:
            raise RuntimeError(f"Failed to reload model {model_id} in classification service")

        return True

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        # Check if model exists
        model = await self._model_repository.find_by_id(model_id)
        if not model:
            return False

        # Don't allow deletion of active models
        active_model = await self._model_repository.find_active_by_name(model.name)
        if active_model and active_model.id == model_id:
            raise ValueError("Cannot delete the currently active model")

        return await self._model_repository.delete(model_id)