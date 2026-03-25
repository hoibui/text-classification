from typing import List
import uuid
from datetime import datetime

from domain.entities.text_classification import TextClassification, ClassificationResult
from domain.services.classification_service import ClassificationService
from domain.repositories.classification_repository import ClassificationRepository
from application.dto.classification_dto import (
    ClassificationRequestDTO,
    ClassificationResponseDTO,
    BatchClassificationRequestDTO
)


class ClassifyTextUseCase:
    """Use case for text classification operations."""

    def __init__(
        self,
        classification_service: ClassificationService,
        classification_repository: ClassificationRepository
    ):
        self._classification_service = classification_service
        self._classification_repository = classification_repository

    async def execute(self, request: ClassificationRequestDTO) -> ClassificationResponseDTO:
        """Execute single text classification."""
        # Create domain entity
        text_classification = TextClassification(
            text=request.text,
            request_id=request.request_id or str(uuid.uuid4()),
            timestamp=datetime.utcnow()
        )

        # Perform classification
        result = await self._classification_service.classify_text(text_classification)

        # Save result for monitoring/analytics
        await self._classification_repository.save_result(result)

        # Convert to DTO and return
        return ClassificationResponseDTO.from_domain(result)

    async def execute_batch(self, request: BatchClassificationRequestDTO) -> List[ClassificationResponseDTO]:
        """Execute batch text classification."""
        # Create domain entities
        text_classifications = [
            TextClassification(
                text=text,
                request_id=f"{request.batch_id or str(uuid.uuid4())}_{i}",
                timestamp=datetime.utcnow()
            )
            for i, text in enumerate(request.texts)
        ]

        # Perform batch classification
        results = await self._classification_service.classify_batch(text_classifications)

        # Save results
        for result in results:
            await self._classification_repository.save_result(result)

        # Convert to DTOs and return
        return [ClassificationResponseDTO.from_domain(result) for result in results]

    async def get_recent_classifications(self, limit: int = 100) -> List[ClassificationResponseDTO]:
        """Get recent classification results."""
        results = await self._classification_repository.get_recent_results(limit)
        return [ClassificationResponseDTO.from_domain(result) for result in results]