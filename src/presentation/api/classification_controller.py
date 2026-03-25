from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging
import time
from datetime import datetime

from application.use_cases.classify_text_use_case import ClassifyTextUseCase
from application.dto.classification_dto import ClassificationRequestDTO, BatchClassificationRequestDTO
from presentation.schemas.classification_schemas import (
    ClassificationRequest,
    BatchClassificationRequest,
    ClassificationResponse
)
from infrastructure.config.dependencies import get_classify_text_use_case

logger = logging.getLogger(__name__)

class ClassificationController:
    """Controller for text classification endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/classify", tags=["Classification"])
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""
        self.router.add_api_route(
            "/",
            self.classify_text,
            methods=["POST"],
            response_model=ClassificationResponse,
            summary="Classify single text",
            description="Classify a single text and return the predicted label with confidence"
        )

        self.router.add_api_route(
            "/batch",
            self.classify_batch,
            methods=["POST"],
            response_model=List[ClassificationResponse],
            summary="Classify multiple texts",
            description="Classify multiple texts in a single request"
        )

    async def classify_text(
        self,
        request: ClassificationRequest,
        use_case: ClassifyTextUseCase = Depends(get_classify_text_use_case)
    ) -> ClassificationResponse:
        """Classify a single text."""
        try:
            start_time = time.time()

            # Convert to DTO
            dto = ClassificationRequestDTO(
                text=request.text,
                return_confidence=request.return_confidence,
                request_id=request.request_id
            )

            # Execute use case
            result = await use_case.execute(dto)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Convert to response schema
            response = ClassificationResponse(
                text=result.text,
                predicted_label=result.predicted_label,
                confidence=result.confidence,
                probabilities=result.probabilities if request.return_confidence else None,
                model_version=result.model_version,
                processing_time=processing_time,
                request_id=result.request_id,
                timestamp=result.timestamp
            )

            logger.info(f"Classified text successfully. Label: {result.predicted_label}, Confidence: {result.confidence:.3f}")
            return response

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    async def classify_batch(
        self,
        request: BatchClassificationRequest,
        use_case: ClassifyTextUseCase = Depends(get_classify_text_use_case)
    ) -> List[ClassificationResponse]:
        """Classify multiple texts in batch."""
        try:
            start_time = time.time()

            # Convert to DTO
            dto = BatchClassificationRequestDTO(
                texts=request.texts,
                return_confidence=request.return_confidence,
                batch_id=request.batch_id
            )

            # Execute use case
            results = await use_case.execute_batch(dto)

            # Calculate processing time per text
            total_processing_time = time.time() - start_time
            per_text_time = total_processing_time / len(request.texts)

            # Convert to response schemas
            responses = []
            for result in results:
                response = ClassificationResponse(
                    text=result.text,
                    predicted_label=result.predicted_label,
                    confidence=result.confidence,
                    probabilities=result.probabilities if request.return_confidence else None,
                    model_version=result.model_version,
                    processing_time=per_text_time,
                    request_id=result.request_id,
                    timestamp=result.timestamp
                )
                responses.append(response)

            logger.info(f"Batch classification completed. {len(request.texts)} texts processed in {total_processing_time:.3f}s")
            return responses

        except Exception as e:
            logger.error(f"Batch classification failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")