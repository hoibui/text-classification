from functools import lru_cache
from typing import Dict, Any

from application.use_cases.classify_text_use_case import ClassifyTextUseCase
from application.use_cases.train_model_use_case import TrainModelUseCase
from application.use_cases.get_model_info_use_case import GetModelInfoUseCase
from domain.services.classification_service import ClassificationService
from domain.services.model_training_service import ModelTrainingService
from domain.repositories.model_repository import ModelRepository
from domain.repositories.classification_repository import ClassificationRepository
from infrastructure.repositories.postgresql_model_repository import PostgreSQLModelRepository
from infrastructure.services.ml_training_service import MLTrainingService
from infrastructure.services.monitoring_service import MonitoringService
from infrastructure.config.settings import Settings


class DependencyContainer:
    """Dependency injection container for the application."""

    def __init__(self):
        self._settings = None
        self._model_repository = None
        self._classification_repository = None
        self._training_service = None
        self._classification_service = None
        self._monitoring_service = None

    @property
    def settings(self) -> Settings:
        """Get application settings."""
        if self._settings is None:
            self._settings = Settings()
        return self._settings

    @property
    def model_repository(self) -> ModelRepository:
        """Get model repository."""
        if self._model_repository is None:
            self._model_repository = PostgreSQLModelRepository(self.settings)
        return self._model_repository

    @property
    def classification_repository(self) -> ClassificationRepository:
        """Get classification repository."""
        # TODO: Implement PostgreSQLClassificationRepository
        if self._classification_repository is None:
            # For now, return a mock or in-memory implementation
            from infrastructure.repositories.in_memory_classification_repository import InMemoryClassificationRepository
            self._classification_repository = InMemoryClassificationRepository()
        return self._classification_repository

    @property
    def training_service(self) -> ModelTrainingService:
        """Get model training service."""
        if self._training_service is None:
            self._training_service = MLTrainingService(self.settings)
        return self._training_service

    @property
    def classification_service(self) -> ClassificationService:
        """Get classification service."""
        if self._classification_service is None:
            from infrastructure.services.ml_classification_service import MLClassificationService
            self._classification_service = MLClassificationService()
        return self._classification_service

    @property
    def monitoring_service(self) -> MonitoringService:
        """Get monitoring service."""
        if self._monitoring_service is None:
            self._monitoring_service = MonitoringService(self.settings)
        return self._monitoring_service

    # Use Cases
    def get_classify_text_use_case(self) -> ClassifyTextUseCase:
        """Get classify text use case."""
        return ClassifyTextUseCase(
            self.classification_service,
            self.classification_repository
        )

    def get_train_model_use_case(self) -> TrainModelUseCase:
        """Get train model use case."""
        return TrainModelUseCase(
            self.training_service,
            self.model_repository
        )

    def get_model_info_use_case(self) -> GetModelInfoUseCase:
        """Get model info use case."""
        return GetModelInfoUseCase(
            self.model_repository,
            self.classification_service
        )


# Global dependency container instance
@lru_cache()
def get_container() -> DependencyContainer:
    """Get the global dependency container."""
    return DependencyContainer()


# FastAPI dependency functions
def get_classify_text_use_case() -> ClassifyTextUseCase:
    """FastAPI dependency for classify text use case."""
    return get_container().get_classify_text_use_case()


def get_train_model_use_case() -> TrainModelUseCase:
    """FastAPI dependency for train model use case."""
    return get_container().get_train_model_use_case()


def get_model_info_use_case() -> GetModelInfoUseCase:
    """FastAPI dependency for model info use case."""
    return get_container().get_model_info_use_case()


def get_monitoring_service() -> MonitoringService:
    """FastAPI dependency for monitoring service."""
    return get_container().monitoring_service