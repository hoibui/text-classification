from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.model import Model, ModelStatus


class ModelRepository(ABC):
    """Repository interface for model persistence."""

    @abstractmethod
    async def save(self, model: Model) -> Model:
        """Save a model and return it with updated ID."""
        pass

    @abstractmethod
    async def find_by_id(self, model_id: str) -> Optional[Model]:
        """Find a model by its ID."""
        pass

    @abstractmethod
    async def find_by_name_and_version(self, name: str, version: str) -> Optional[Model]:
        """Find a model by name and version."""
        pass

    @abstractmethod
    async def find_active_by_name(self, name: str) -> Optional[Model]:
        """Find the currently active model by name."""
        pass

    @abstractmethod
    async def find_by_status(self, status: ModelStatus) -> List[Model]:
        """Find all models with a specific status."""
        pass

    @abstractmethod
    async def list_all(self, limit: Optional[int] = None) -> List[Model]:
        """List all models with optional limit."""
        pass

    @abstractmethod
    async def update(self, model: Model) -> Model:
        """Update an existing model."""
        pass

    @abstractmethod
    async def delete(self, model_id: str) -> bool:
        """Delete a model by ID. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    async def set_active(self, model_id: str, name: str) -> bool:
        """Set a model as the active one for a given name."""
        pass