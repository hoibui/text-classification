# Charge Type Classification System

A professional, scalable ML system for classifying utility charge types with comprehensive MLOps capabilities.

## Features

- **Multiple Model Support**: Traditional ML (Logistic Regression, Random Forest, SVM, etc.) and Transformer models
- **Production-Ready API**: FastAPI-based REST API with health checks and monitoring
- **MLOps Pipeline**: Model versioning, experiment tracking, and automated retraining
- **Monitoring & Observability**: Prometheus metrics, logging, and data drift detection
- **Containerized Deployment**: Docker and Docker Compose setup
- **CI/CD Integration**: GitHub Actions workflows for testing and deployment

## Quick Start

### 1. Setup Database

Ensure PostgreSQL is running on localhost:5432 with database 'mlops', user 'hoibui', password 'admin'.

```bash
python setup_database.py
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/trainer.py
```

### 4. Start the API Server

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8001
```

### 5. Make Predictions

```bash
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Peak energy usage charge"}'
```

## Docker Deployment

### Build and Run

```bash
docker-compose up -d
```

This starts:
- Charge Classification API (port 8000)
- MLflow Tracking Server (port 5000)
- Prometheus Monitoring (port 9090)
- Grafana Dashboard (port 3000)

### Access Services

- API Documentation: http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## API Endpoints

### Single Prediction
```bash
POST /predict
{
  "text": "Network connection fee",
  "return_confidence": true
}
```

### Batch Predictions
```bash
POST /predict/batch
{
  "texts": ["Peak energy charge", "Network fee"],
  "return_confidence": true
}
```

### Health Check
```bash
GET /health
```

### Model Information
```bash
GET /model/info
```

### Metrics
```bash
GET /metrics
```

## Configuration

Edit `config.yaml` to customize:

- Model parameters
- Training settings
- MLOps configuration
- Deployment settings

## Model Registry

The system includes a model registry for version management:

```python
from src.model_registry import ModelRegistry

registry = ModelRegistry()

# Register a new model
model_id = registry.register_model(
    name="charge_classifier",
    version="1.1.0",
    file_path="models/new_model.pkl",
    performance_metrics={"accuracy": 0.96},
    make_active=True
)

# Load active model
model = registry.load_active_model("charge_classifier")
```

## Monitoring

The system includes comprehensive monitoring:

- **Performance Metrics**: Response times, accuracy, error rates
- **System Metrics**: CPU, memory, disk usage
- **Data Drift Detection**: Automatic detection of data distribution changes
- **Health Checks**: API and model health status

## CI/CD Pipeline

GitHub Actions workflows provide:

- **Testing**: Automated testing on push/PR
- **Model Validation**: Validates model training pipeline
- **Docker Build**: Builds and pushes container images
- **Scheduled Retraining**: Weekly automatic model retraining

## Project Structure

```
classification/
├── src/
│   ├── __init__.py
│   ├── trainer.py          # Model training pipeline
│   ├── api.py              # FastAPI application
│   ├── monitor.py          # Monitoring and logging
│   └── model_registry.py   # Model versioning
├── data/
│   └── train.csv           # Training data
├── models/                 # Saved models
├── logs/                   # Log files
├── monitoring/
│   └── prometheus.yml      # Prometheus configuration
├── .github/
│   └── workflows/          # CI/CD pipelines
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container image
├── docker-compose.yml     # Multi-service deployment
└── README.md              # This file
```

## Data Format

Training data should be in CSV format with two columns:
- Column 1: Label (charge type)
- Column 2: Text description

Example:
```csv
ENERGY_LINE_ITEMS,Peak energy usage charge
NETWORK_SERVICE,Network connection fee
DISCOUNTS,Customer discount applied
```

## Model Performance

The system automatically evaluates multiple algorithms and selects the best performer:

- **Traditional ML Models**: Logistic Regression, Random Forest, SVM, Naive Bayes
- **Deep Learning**: DistilBERT transformer model
- **Feature Engineering**: TF-IDF word/character n-grams, Count vectors
- **Evaluation**: Cross-validation, comprehensive metrics reporting

## Security

- Input validation and sanitization
- No credential exposure in logs
- Container security best practices
- Health check endpoints for monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.