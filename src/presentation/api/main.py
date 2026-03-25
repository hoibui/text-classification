from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import logging

from presentation.schemas.classification_schemas import HealthResponse
from presentation.api.classification_controller import ClassificationController
from infrastructure.config.dependencies import get_monitoring_service
from infrastructure.services.monitoring_service import MonitoringService
from infrastructure.config.settings import Settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Load settings
    settings = Settings()

    # Create FastAPI app
    app = FastAPI(
        title="Text Classification API",
        description="Professional API for text classification with Clean Architecture",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Include routers
    classification_controller = ClassificationController()
    app.include_router(classification_controller.router)

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check(
        monitoring_service: MonitoringService = Depends(get_monitoring_service)
    ):
        """Health check endpoint."""
        health_status = monitoring_service.get_health_status()

        return HealthResponse(
            status=health_status["status"],
            timestamp=health_status["timestamp"],
            version="2.0.0",
            model_loaded=True,  # TODO: Check actual model status
            database_connected=True  # TODO: Check actual DB status
        )

    # Metrics endpoint for Prometheus
    @app.get("/metrics", tags=["Monitoring"])
    async def get_metrics(
        monitoring_service: MonitoringService = Depends(get_monitoring_service)
    ):
        """Prometheus metrics endpoint."""
        metrics_data = monitoring_service.generate_metrics()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

    return app


# Create the app instance
app = create_app()