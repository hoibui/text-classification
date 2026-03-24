import os
import json
import psycopg2
import psycopg2.extras
import joblib
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

class ModelRegistry:
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "mlops",
                 user: str = "hoibui",
                 password: str = "admin"):
        self.db_config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def _init_database(self):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS custom_models (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version VARCHAR(100) NOT NULL,
                    file_path TEXT NOT NULL,
                    metadata JSONB,
                    performance_metrics JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT FALSE,
                    model_hash VARCHAR(32),
                    UNIQUE(name, version)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS custom_experiments (
                    id SERIAL PRIMARY KEY,
                    experiment_name VARCHAR(255) NOT NULL,
                    model_id INTEGER REFERENCES custom_models(id),
                    parameters JSONB,
                    metrics JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_custom_models_name ON custom_models(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_custom_models_active ON custom_models(name, is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_custom_experiments_model ON custom_experiments(model_id)')

            conn.commit()
            conn.close()
            self.logger.info("Database initialized successfully")
        except psycopg2.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    def _calculate_model_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of model file for integrity checking"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def register_model(self,
                      name: str,
                      version: str,
                      file_path: str,
                      metadata: Dict[str, Any] = None,
                      performance_metrics: Dict[str, float] = None,
                      make_active: bool = False) -> int:
        """Register a new model in the registry"""

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        model_hash = self._calculate_model_hash(file_path)

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Deactivate current active model if making this one active
            if make_active:
                cursor.execute(
                    "UPDATE custom_models SET is_active = FALSE WHERE name = %s AND is_active = TRUE",
                    (name,)
                )

            # Insert new model
            cursor.execute('''
                INSERT INTO custom_models (name, version, file_path, metadata, performance_metrics, is_active, model_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            ''', (
                name,
                version,
                file_path,
                json.dumps(metadata or {}),
                json.dumps(performance_metrics or {}),
                make_active,
                model_hash
            ))

            model_id = cursor.fetchone()[0]
            conn.commit()

            self.logger.info(f"Model registered: {name} v{version} (ID: {model_id})")
            return model_id

        except psycopg2.IntegrityError:
            raise ValueError(f"Model {name} version {version} already exists")
        finally:
            conn.close()

    def get_model(self, name: str, version: str = None) -> Optional[Dict[str, Any]]:
        """Get model information by name and version"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if version:
            cursor.execute(
                "SELECT * FROM custom_models WHERE name = %s AND version = %s",
                (name, version)
            )
        else:
            # Get latest version
            cursor.execute(
                "SELECT * FROM custom_models WHERE name = %s ORDER BY created_at DESC LIMIT 1",
                (name,)
            )

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_active_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the currently active model"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute(
            "SELECT * FROM custom_models WHERE name = %s AND is_active = TRUE",
            (name,)
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def list_models(self, name: str = None) -> List[Dict[str, Any]]:
        """List all models or models with specific name"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        if name:
            cursor.execute(
                "SELECT * FROM custom_models WHERE name = %s ORDER BY created_at DESC",
                (name,)
            )
        else:
            cursor.execute("SELECT * FROM custom_models ORDER BY created_at DESC")

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def promote_model(self, name: str, version: str) -> bool:
        """Promote a model version to active status"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Deactivate all models with this name
            cursor.execute(
                "UPDATE custom_models SET is_active = FALSE WHERE name = %s",
                (name,)
            )

            # Activate the specified version
            cursor.execute(
                "UPDATE custom_models SET is_active = TRUE WHERE name = %s AND version = %s",
                (name, version)
            )

            if cursor.rowcount == 0:
                return False

            conn.commit()
            self.logger.info(f"Model {name} v{version} promoted to active")
            return True

        finally:
            conn.close()

    def load_active_model(self, name: str):
        """Load the active model from disk"""
        model_info = self.get_active_model(name)
        if not model_info:
            raise ValueError(f"No active model found for {name}")

        file_path = model_info['file_path']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        # Verify model integrity
        current_hash = self._calculate_model_hash(file_path)
        if current_hash != model_info['model_hash']:
            self.logger.warning(f"Model file hash mismatch for {name}")

        return joblib.load(file_path)

    def log_experiment(self,
                      experiment_name: str,
                      model_id: int,
                      parameters: Dict[str, Any],
                      metrics: Dict[str, float]) -> int:
        """Log an experiment"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO custom_experiments (experiment_name, model_id, parameters, metrics)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        ''', (
            experiment_name,
            model_id,
            json.dumps(parameters),
            json.dumps(metrics)
        ))

        experiment_id = cursor.fetchone()[0]
        conn.commit()
        conn.close()

        self.logger.info(f"Experiment logged: {experiment_name} (ID: {experiment_id})")
        return experiment_id

    def get_model_experiments(self, model_id: int) -> List[Dict[str, Any]]:
        """Get all experiments for a model"""
        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cursor.execute(
            "SELECT * FROM custom_experiments WHERE model_id = %s ORDER BY created_at DESC",
            (model_id,)
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def compare_models(self, name: str, metric: str = 'accuracy') -> List[Dict[str, Any]]:
        """Compare models by a specific metric"""
        models = self.list_models(name)

        comparison = []
        for model in models:
            metrics = model['performance_metrics']
            if metric in metrics:
                comparison.append({
                    'version': model['version'],
                    'metric_value': metrics[metric],
                    'created_at': model['created_at'],
                    'is_active': model['is_active']
                })

        # Sort by metric value (descending)
        comparison.sort(key=lambda x: x['metric_value'], reverse=True)
        return comparison

if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()

    # Register a model
    model_id = registry.register_model(
        name="charge_classifier",
        version="1.0.0",
        file_path="models/best_charge_classifier.pkl",
        metadata={"training_data": "data/train.csv", "algorithm": "logistic_regression"},
        performance_metrics={"accuracy": 0.95, "f1_score": 0.94},
        make_active=True
    )

    # List all models
    models = registry.list_models("charge_classifier")
    print(f"Found {len(models)} models")

    # Get active model
    active_model = registry.get_active_model("charge_classifier")
    if active_model:
        print(f"Active model: {active_model['version']} (accuracy: {active_model['performance_metrics']['accuracy']})")