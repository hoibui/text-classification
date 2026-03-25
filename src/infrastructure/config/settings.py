import os
import yaml
import string
from dataclasses import dataclass
from typing import Dict, Any, Optional
from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    run_name: str


@dataclass
class ModelConfig:
    save_path: str
    name_prefix: str


@dataclass
class APIConfig:
    host: str
    port: int
    workers: int


@dataclass
class MonitoringConfig:
    enabled: bool
    log_level: str
    metrics_port: int


@dataclass
class TrainingConfig:
    data_path: str
    test_size: float
    random_state: int
    cv_folds: int


class Settings:
    """Application settings loaded from environment variables and config files."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Try to find config.yaml in the project root
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            config_path = os.path.join(project_root, "config.yaml")
        # Load environment variables
        load_dotenv()

        # Load and substitute environment variables in config
        with open(config_path, 'r') as file:
            config_content = file.read()

        # Substitute environment variables
        config_content = string.Template(config_content).safe_substitute(os.environ)
        self._config = yaml.safe_load(config_content)

        # Initialize configurations
        self._init_database_config()
        self._init_mlflow_config()
        self._init_model_config()
        self._init_api_config()
        self._init_monitoring_config()
        self._init_training_config()

    def _init_database_config(self):
        """Initialize database configuration."""
        self.database = DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            name=os.getenv('DB_NAME', 'mlops'),
            user=os.getenv('DB_USER', 'admin'),
            password=os.getenv('DB_PASSWORD', 'admin')
        )

    def _init_mlflow_config(self):
        """Initialize MLflow configuration."""
        self.mlflow = MLflowConfig(
            tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlruns.db'),
            experiment_name=os.getenv('MLFLOW_EXPERIMENT_NAME', 'text_classification'),
            run_name=os.getenv('MLFLOW_RUN_NAME', 'text_classification_training')
        )

    def _init_model_config(self):
        """Initialize model configuration."""
        self.model = ModelConfig(
            save_path=os.getenv('MODEL_SAVE_PATH', 'models'),
            name_prefix=os.getenv('MODEL_NAME_PREFIX', 'best_model')
        )

    def _init_api_config(self):
        """Initialize API configuration."""
        self.api = APIConfig(
            host=os.getenv('API_HOST', '0.0.0.0'),
            port=int(os.getenv('API_PORT', 8000)),
            workers=int(os.getenv('API_WORKERS', 4))
        )

    def _init_monitoring_config(self):
        """Initialize monitoring configuration."""
        self.monitoring = MonitoringConfig(
            enabled=os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
            log_level=os.getenv('MONITORING_LOG_LEVEL', 'INFO'),
            metrics_port=int(os.getenv('MONITORING_METRICS_PORT', 8080))
        )

    def _init_training_config(self):
        """Initialize training configuration."""
        self.training = TrainingConfig(
            data_path=os.getenv('DATA_PATH', 'data/train.csv'),
            test_size=float(os.getenv('TEST_SIZE', 0.2)),
            random_state=int(os.getenv('RANDOM_STATE', 42)),
            cv_folds=int(os.getenv('CV_FOLDS', 5))
        )

    @property
    def config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self._config

    @property
    def project_root(self) -> str:
        """Get project root directory."""
        return os.getenv('PROJECT_ROOT', os.getcwd())

    @property
    def log_level(self) -> str:
        """Get logging level."""
        return os.getenv('LOG_LEVEL', 'INFO')

    @property
    def log_file(self) -> str:
        """Get log file path."""
        return os.getenv('LOG_FILE', 'logs/training.log')