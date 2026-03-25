import logging
import psutil
import time
from datetime import datetime
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from infrastructure.config.settings import Settings


class MonitoringService:
    """Infrastructure service for monitoring and metrics collection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # Prometheus metrics
        self.request_counter = Counter(
            'classification_requests_total',
            'Total number of classification requests',
            ['method', 'endpoint', 'status']
        )

        self.request_duration = Histogram(
            'classification_request_duration_seconds',
            'Classification request duration in seconds',
            ['method', 'endpoint']
        )

        self.model_predictions = Counter(
            'model_predictions_total',
            'Total number of model predictions',
            ['model_name', 'predicted_label']
        )

        self.model_confidence = Histogram(
            'model_confidence_score',
            'Model confidence scores',
            ['model_name']
        )

        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )

        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes'
        )

        self.active_connections = Gauge(
            'database_active_connections',
            'Number of active database connections'
        )

    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record API request metrics."""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_prediction(self, model_name: str, predicted_label: str, confidence: float):
        """Record model prediction metrics."""
        self.model_predictions.labels(model_name=model_name, predicted_label=predicted_label).inc()
        self.model_confidence.labels(model_name=model_name).observe(confidence)

    def update_system_metrics(self):
        """Update system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu_usage.set(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.system_memory_usage.set(memory.used)

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        return {
            "status": "healthy" if cpu_percent < 80 and memory.percent < 80 else "warning",
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": datetime.utcnow().isoformat()
        }

    def generate_metrics(self) -> str:
        """Generate Prometheus metrics."""
        self.update_system_metrics()
        return generate_latest()