# Text Classification System

A professional, scalable ML system for text classification with comprehensive MLOps capabilities.

## Features

- **Multiple Model Support**: Traditional ML (Logistic Regression, Random Forest, SVM, etc.) and Transformer models
- **Production-Ready API**: FastAPI-based REST API with health checks and monitoring
- **MLOps Pipeline**: Model versioning, experiment tracking, and automated retraining
- **Monitoring & Observability**: Prometheus metrics, logging, and data drift detection
- **Containerized Deployment**: Docker and Docker Compose setup
- **CI/CD Integration**: GitHub Actions workflows for testing and deployment

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd text-classification

# Run automated setup (creates virtual environment and .env file)
./setup.sh
```

### 2. Configure Environment

Edit the `.env` file with your specific settings:

```bash
# Update these values in .env:
PROJECT_ROOT=/your/path/to/text-classification
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mlops
DB_USER=your_username
DB_PASSWORD=your_password
MLFLOW_TRACKING_URI=postgresql://your_username:your_password@localhost:5432/mlflow_tracking
```

### 3. Setup Database

```bash
python setup_database.py
```

### 4. Train the Model

```bash
./run_trainer.sh
# or activate environment first:
# source venv/bin/activate && python src/trainer.py
```

### 5. Start the API Server

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8001
```

### 6. Make Predictions

```bash
curl -X POST "http://localhost:8001/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your text to classify here"}'
```

## Docker Deployment

### Prerequisites
First complete the setup steps to create your `.env` file with proper configuration.

### Build and Run

```bash
# Make sure .env file is configured first
docker-compose up -d
```

This starts:
- Text Classification API (port from API_PORT env var, default 8000)
- MLflow Tracking Server (port 5000)
- Prometheus Monitoring (port 9090)
- Grafana Dashboard (port 3000)

### Local Development with Docker

For development with hot reloading and source code mounting:

```bash
# Copy and customize the override file
cp docker-compose.override.yml.example docker-compose.override.yml

# Edit docker-compose.override.yml and uncomment the development settings
# Then start with the override
docker-compose up -d
```

### Access Services

- API Documentation: http://localhost:${API_PORT}/docs (default: http://localhost:8000/docs)
- MLflow UI: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## API Endpoints

### Single Prediction
```bash
POST /predict
{
  "text": "Your text to classify",
  "return_confidence": true
}
```

### Batch Predictions
```bash
POST /predict/batch
{
  "texts": ["First text to classify", "Second text to classify"],
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

The system uses environment variables for configuration:

### Environment Variables (.env file)

- **PROJECT_ROOT**: Path to your project directory
- **DB_HOST/DB_PORT/DB_USER/DB_PASSWORD**: Database connection settings
- **MLFLOW_TRACKING_URI**: MLflow tracking server URL
- **MLFLOW_EXPERIMENT_NAME**: Name for MLflow experiment
- **MODEL_SAVE_PATH**: Directory to save trained models
- **API_HOST/API_PORT**: API server configuration
- **LOG_LEVEL**: Logging level (INFO, DEBUG, etc.)

### Configuration Files

- `.env`: Your personal environment configuration (not in git)
- `.env.example`: Template file for team members
- `config.yaml`: Model parameters and training settings (uses environment variables)

## Model Registry

The system includes a model registry for version management:

```python
from src.model_registry import ModelRegistry

registry = ModelRegistry()

# Register a new model
model_id = registry.register_model(
    name="text_classifier",
    version="1.1.0",
    file_path="models/new_model.pkl",
    performance_metrics={"accuracy": 0.96},
    make_active=True
)

# Load active model
model = registry.load_active_model("text_classifier")
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
text-classification/
├── src/
│   ├── __init__.py
│   ├── trainer.py          # Model training pipeline
│   ├── api.py              # FastAPI application
│   ├── monitor.py          # Monitoring and logging
│   └── model_registry.py   # Model versioning
├── data/
│   └── train.csv           # Training data
├── models/                 # Saved models (excluded from git)
├── logs/                   # Log files (excluded from git)
├── venv/                   # Virtual environment (excluded from git)
├── monitoring/
│   └── prometheus.yml      # Prometheus configuration
├── .github/
│   └── workflows/          # CI/CD pipelines
├── .env                    # Environment variables (excluded from git)
├── .env.example            # Environment template (included in git)
├── config.yaml             # Configuration file (uses env variables)
├── requirements.txt        # Python dependencies
├── setup.sh                # Automated setup script
├── run_trainer.sh          # Script to run trainer
├── Dockerfile              # Container image
├── docker-compose.yml      # Multi-service deployment
├── docker-compose.override.yml        # Local dev overrides (excluded from git)
├── docker-compose.override.yml.example # Override template (included in git)
└── README.md               # This file
```

## Data Format

Training data should be in CSV format with two columns:
- Column 1: Label (classification category)
- Column 2: Text description

Example:
```csv
CATEGORY_A,Text sample for category A
CATEGORY_B,Text sample for category B
CATEGORY_C,Text sample for category C
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