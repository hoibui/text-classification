from typing import List, Optional
from datetime import datetime

from domain.entities.text_classification import ClassificationResult
from domain.repositories.classification_repository import ClassificationRepository


class InMemoryClassificationRepository(ClassificationRepository):
    """In-memory implementation of ClassificationRepository for development."""

    def __init__(self):
        self._results = []

    async def save_result(self, result: ClassificationResult) -> ClassificationResult:
        """Save a classification result."""
        self._results.append(result)
        return result

    async def find_by_request_id(self, request_id: str) -> Optional[ClassificationResult]:
        """Find a classification result by request ID."""
        for result in self._results:
            if result.request_id == request_id:
                return result
        return None

    async def find_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[ClassificationResult]:
        """Find classification results within a time range."""
        filtered_results = [
            result for result in self._results
            if result.timestamp and start_time <= result.timestamp <= end_time
        ]

        if limit:
            return filtered_results[:limit]
        return filtered_results

    async def get_recent_results(self, limit: int = 100) -> List[ClassificationResult]:
        """Get the most recent classification results."""
        sorted_results = sorted(
            self._results,
            key=lambda x: x.timestamp or datetime.min,
            reverse=True
        )
        return sorted_results[:limit]

    async def count_by_label(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict:
        """Count classification results by predicted label within time range."""
        filtered_results = self._results

        if start_time or end_time:
            filtered_results = [
                result for result in self._results
                if result.timestamp and
                (not start_time or result.timestamp >= start_time) and
                (not end_time or result.timestamp <= end_time)
            ]

        counts = {}
        for result in filtered_results:
            label = result.predicted_label
            counts[label] = counts.get(label, 0) + 1

        return counts

    async def get_performance_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> dict:
        """Get aggregated performance metrics for the time range."""
        filtered_results = self._results

        if start_time or end_time:
            filtered_results = [
                result for result in self._results
                if result.timestamp and
                (not start_time or result.timestamp >= start_time) and
                (not end_time or result.timestamp <= end_time)
            ]

        if not filtered_results:
            return {}

        # Calculate basic metrics
        total_predictions = len(filtered_results)
        avg_confidence = sum(r.confidence for r in filtered_results) / total_predictions
        processing_times = [r.processing_time for r in filtered_results if r.processing_time]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        return {
            "total_predictions": total_predictions,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "unique_labels": len(set(r.predicted_label for r in filtered_results))
        }