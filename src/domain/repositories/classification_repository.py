from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from ..entities.text_classification import ClassificationResult


class ClassificationRepository(ABC):
    """Repository interface for classification result persistence."""

    @abstractmethod
    async def save_result(self, result: ClassificationResult) -> ClassificationResult:
        """Save a classification result."""
        pass

    @abstractmethod
    async def find_by_request_id(self, request_id: str) -> Optional[ClassificationResult]:
        """Find a classification result by request ID."""
        pass

    @abstractmethod
    async def find_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[ClassificationResult]:
        """Find classification results within a time range."""
        pass

    @abstractmethod
    async def get_recent_results(self, limit: int = 100) -> List[ClassificationResult]:
        """Get the most recent classification results."""
        pass

    @abstractmethod
    async def count_by_label(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict:
        """Count classification results by predicted label within time range."""
        pass

    @abstractmethod
    async def get_performance_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict:
        """Get aggregated performance metrics for the time range."""
        pass