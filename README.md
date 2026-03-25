# Text Classification System 🚀

A professional, scalable ML system for text classification built with **Clean Architecture** and comprehensive MLOps capabilities.

## ✨ Features

### 🏗️ **Clean Architecture**
- **Domain-Driven Design**: Business logic separated from infrastructure concerns
- **SOLID Principles**: Maintainable, testable, and extensible codebase
- **Dependency Injection**: Loosely coupled components for easy testing and swapping
- **Layered Architecture**: Domain, Application, Infrastructure, and Presentation layers

### 🤖 **Machine Learning**
- **Traditional ML Models**: Logistic Regression, Random Forest, SVM, Naive Bayes, Gradient Boosting
- **Advanced Feature Engineering**: TF-IDF word/character n-grams with optimal parameters
- **Model Registry**: Version management and model lifecycle tracking
- **Automated Training**: CLI-based training with configurable pipelines
- **Performance Monitoring**: Comprehensive metrics and model evaluation (98%+ accuracy)

### 🚀 **Production-Ready**
- **FastAPI REST API**: High-performance API with automatic documentation
- **Health Checks**: System and model health monitoring
- **Prometheus Metrics**: Comprehensive observability and monitoring
- **Containerized Deployment**: Docker and Docker Compose support

### 🔧 **MLOps & DevOps**
- **Environment-Driven Config**: All settings via environment variables
- **Database Integration**: PostgreSQL for model registry and metrics
- **MLflow Integration**: Experiment tracking and model versioning
- **Team-Friendly Setup**: One-command setup for all team members

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
# Train with default settings
./run_trainer.sh

# Train with custom parameters
./run_trainer.sh custom_model 2.0.0 data/train.csv

# Advanced training options
python src/main.py train --name custom_model --version 1.0.0 --data-path data/custom_train.csv
```

### 5. Start the API Server

```bash
# Start API server with default settings
./run_api.sh

# Or with custom settings
python src/main.py serve --host 0.0.0.0 --port 8001 --workers 4
```

### 6. Make Predictions

```bash
# Single prediction
curl -X POST "http://localhost:8001/classify/" \
     -H "Content-Type: application/json" \
     -d '{"text": "renewable energy certificate costs", "return_confidence": true}'

# Expected response:
# {
#   "text": "renewable energy certificate costs",
#   "predicted_label": "CERTIFICATES",
#   "confidence": 0.41,
#   "probabilities": {"CERTIFICATES": 0.41, "GREEN": 0.16, ...},
#   "model_version": "tfidf_char_random_forest-best_model_...",
#   "processing_time": 0.045
# }
```

### 7. Model Management

```bash
# List all models
python src/main.py list

# List models by status
python src/main.py list --status ready
```

## 🐳 Docker Deployment

### Prerequisites
First complete the setup steps to create your `.env` file with proper configuration.

### Quick Start with Docker

```bash
# 1. Make sure .env file is configured
./setup.sh

# 2. Start all services (API + Database + Monitoring)
docker-compose up -d

# 3. Train a model in Docker
docker-compose exec text-classifier ./run_trainer.sh

# 4. Check running services
docker-compose ps
```

### Services Started

```bash
docker-compose up -d
```

This starts:
- **Text Classification API** (port from API_PORT env var, default 8001)
- **PostgreSQL Database** (port 5432) - for model registry
- **MLflow Tracking Server** (port 5000) - experiment tracking
- **Prometheus Monitoring** (port 9090) - metrics collection

### Access Services

- API Documentation: http://localhost:8001/docs
- MLflow UI: http://localhost:5000
- Prometheus: http://localhost:9090

## 🔌 API Endpoints

The Clean Architecture API provides the following endpoints:

### 📝 Text Classification

#### Single Text Classification
```bash
POST /classify/
{
  "text": "Your text to classify here",
  "return_confidence": true,
  "request_id": "optional-request-id"
}
```

**Response:**
```json
{
  "text": "Your text to classify here",
  "predicted_label": "CATEGORY_A",
  "confidence": 0.95,
  "probabilities": {
    "CATEGORY_A": 0.95,
    "CATEGORY_B": 0.04,
    "CATEGORY_C": 0.01
  },
  "model_version": "tfidf_char_random_forest-best_model_20260325_091723",
  "processing_time": 0.023,
  "request_id": "optional-request-id",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Batch Text Classification
```bash
POST /classify/batch
{
  "texts": ["First text", "Second text", "Third text"],
  "return_confidence": true,
  "batch_id": "optional-batch-id"
}
```

### 🏥 System Health

#### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "2.0.0",
  "model_loaded": true,
  "database_connected": true
}
```

### 📊 Monitoring

#### Prometheus Metrics
```bash
GET /metrics
```

Returns Prometheus-formatted metrics for monitoring and alerting.

### 📚 Interactive Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## Configuration

The system uses environment variables for configuration:

### Environment Variables (.env file)

- **PROJECT_ROOT**: Path to your project directory
- **DB_HOST/DB_PORT/DB_USER/DB_PASSWORD**: Database connection settings
- **MLFLOW_TRACKING_URI**: MLflow tracking server URL
- **MLFLOW_EXPERIMENT_NAME**: Name for MLflow experiment
- **MODEL_SAVE_PATH**: Directory to save trained models
- **API_HOST/API_PORT**: API server configuration (default port 8001)
- **LOG_LEVEL**: Logging level (INFO, DEBUG, etc.)

### Configuration Files

- `.env`: Your personal environment configuration (not in git)
- `.env.example`: Template file for team members
- `config.yaml`: Model parameters and training settings

## Model Performance

The system automatically evaluates multiple traditional ML algorithms and selects the best performer:

### 🎯 **Current Performance (98.24% Accuracy)**
- **Best Model**: TF-IDF Character-level Random Forest
- **Feature Engineering**: Character n-grams (1-4) with TF-IDF weighting
- **Cross-Validation**: Stratified K-fold validation
- **Metrics**: Comprehensive accuracy, precision, recall, F1-score reporting

### 🔄 **Algorithms Evaluated**
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble method with multiple decision trees
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Naive Bayes**: Probabilistic classification
- **Gradient Boosting**: Sequential ensemble learning

### 📊 **Feature Engineering**
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **Character N-grams**: 1-4 character sequences for better pattern recognition
- **Word N-grams**: 1-2 word sequences for semantic understanding
- **Automatic Feature Selection**: Best performing features selected automatically

## 🏗️ Clean Architecture Structure

This project follows **Clean Architecture** principles with clear separation of concerns:

```
src/
├── domain/                 # 🔵 Domain Layer (Business Logic)
│   ├── entities/          # Business entities (Model, Classification)
│   ├── repositories/      # Repository interfaces
│   ├── services/         # Domain service interfaces
│   └── value_objects/    # Value objects (Confidence, Version)
├── application/           # 🟢 Application Layer (Use Cases)
│   ├── use_cases/        # Business use cases
│   └── dto/              # Data transfer objects
├── infrastructure/       # 🟡 Infrastructure Layer (External Concerns)
│   ├── config/          # Configuration and DI container
│   ├── repositories/    # Database implementations
│   └── services/        # ML and monitoring services
├── presentation/         # 🟠 Presentation Layer (API)
│   ├── api/             # REST API controllers
│   └── schemas/         # Request/response schemas
└── main.py              # CLI entry point and orchestration
```

### 📁 Project Structure

```
text-classification/
├── src/                    # Clean Architecture source code
│   ├── domain/            # Business logic (framework-independent)
│   ├── application/       # Use cases and DTOs
│   ├── infrastructure/    # External services and data access
│   ├── presentation/      # API controllers and schemas
│   └── main.py           # CLI interface
├── data/                  # Training and test data
├── models/               # Saved models (excluded from git)
├── logs/                 # Log files (excluded from git)
├── mlruns/              # MLflow experiment tracking
├── monitoring/           # Monitoring and observability
├── .env                 # Environment variables (excluded from git)
├── .env.example         # Environment template
├── config.yaml          # Model training configuration
├── requirements.txt     # Python dependencies
├── setup.sh            # One-command setup script
├── run_trainer.sh      # Training script
├── run_api.sh          # API server script
├── list_models.sh      # Model management script
├── Dockerfile          # Container image
├── docker-compose.yml  # Multi-service deployment
└── README.md           # This documentation
```

## 🎯 Clean Architecture Benefits

### ✅ **Testability**
- **Unit Tests**: Mock dependencies easily with interfaces
- **Integration Tests**: Test layers in isolation
- **End-to-End Tests**: Test complete user workflows

### ✅ **Maintainability**
- **Clear Boundaries**: Each layer has single responsibility
- **Loose Coupling**: Easy to modify without breaking other parts
- **Dependency Inversion**: Business logic doesn't depend on frameworks

### ✅ **Scalability**
- **Pluggable Architecture**: Swap ML frameworks or databases easily
- **Team Scaling**: Multiple developers can work on different layers
- **Feature Addition**: Add new use cases without touching existing code

### ✅ **Framework Independence**
- **ML Framework Agnostic**: Easy to switch between scikit-learn implementations
- **Database Agnostic**: Switch from PostgreSQL to other databases easily
- **API Framework Agnostic**: Replace FastAPI with other frameworks if needed

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

## Monitoring

The system includes comprehensive monitoring:

- **Performance Metrics**: Response times, accuracy, error rates
- **System Metrics**: CPU, memory, disk usage
- **Model Metrics**: Classification performance and drift detection
- **Health Checks**: API and model health status
- **MLflow Tracking**: Experiment logging and model versioning

## Security

- Input validation and sanitization
- No credential exposure in logs
- Container security best practices
- Health check endpoints for monitoring
- Environment-based configuration for sensitive data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.