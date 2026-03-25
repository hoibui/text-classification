import psycopg2
import psycopg2.extras
import json
import hashlib
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from domain.entities.model import Model, ModelMetrics, ModelStatus
from domain.repositories.model_repository import ModelRepository
from infrastructure.config.settings import Settings


class PostgreSQLModelRepository(ModelRepository):
    """PostgreSQL implementation of ModelRepository."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._init_database()

    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.settings.database.host,
            port=self.settings.database.port,
            database=self.settings.database.name,
            user=self.settings.database.user,
            password=self.settings.database.password
        )

    def _init_database(self):
        """Initialize database tables."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version VARCHAR(100) NOT NULL,
                    model_type VARCHAR(100) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    file_path TEXT,
                    metadata JSONB,
                    accuracy FLOAT,
                    f1_score FLOAT,
                    precision_score FLOAT,
                    recall_score FLOAT,
                    additional_metrics JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT FALSE,
                    model_hash VARCHAR(32),
                    UNIQUE(name, version)
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    def _model_from_row(self, row: dict) -> Model:
        """Convert database row to Model entity."""
        metrics = None
        if row['accuracy'] is not None and row['f1_score'] is not None:
            metrics = ModelMetrics(
                accuracy=row['accuracy'],
                f1_score=row['f1_score'],
                precision=row['precision_score'],
                recall=row['recall_score'],
                additional_metrics=row['additional_metrics'] or {}
            )

        return Model(
            id=str(row['id']),
            name=row['name'],
            version=row['version'],
            model_type=row['model_type'],
            status=ModelStatus(row['status']),
            metrics=metrics,
            metadata=row['metadata'] or {},
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            file_path=row['file_path']
        )

    async def save(self, model: Model) -> Model:
        """Save a model."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute('''
                INSERT INTO models (
                    name, version, model_type, status, file_path, metadata,
                    accuracy, f1_score, precision_score, recall_score,
                    additional_metrics, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            ''', (
                model.name,
                model.version,
                model.model_type,
                model.status.value,
                model.file_path,
                json.dumps(model.metadata),
                model.metrics.accuracy if model.metrics else None,
                model.metrics.f1_score if model.metrics else None,
                model.metrics.precision if model.metrics else None,
                model.metrics.recall if model.metrics else None,
                json.dumps(model.metrics.additional_metrics) if model.metrics else None,
                model.created_at,
                model.updated_at
            ))

            row = cursor.fetchone()
            conn.commit()
            return self._model_from_row(row)
        finally:
            conn.close()

    async def find_by_id(self, model_id: str) -> Optional[Model]:
        """Find model by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('SELECT * FROM models WHERE id = %s', (int(model_id),))
            row = cursor.fetchone()
            return self._model_from_row(row) if row else None
        finally:
            conn.close()

    async def find_by_name_and_version(self, name: str, version: str) -> Optional[Model]:
        """Find model by name and version."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                'SELECT * FROM models WHERE name = %s AND version = %s',
                (name, version)
            )
            row = cursor.fetchone()
            return self._model_from_row(row) if row else None
        finally:
            conn.close()

    async def find_active_by_name(self, name: str) -> Optional[Model]:
        """Find active model by name."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                'SELECT * FROM models WHERE name = %s AND is_active = true',
                (name,)
            )
            row = cursor.fetchone()
            return self._model_from_row(row) if row else None
        finally:
            conn.close()

    async def find_by_status(self, status: ModelStatus) -> List[Model]:
        """Find models by status."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('SELECT * FROM models WHERE status = %s', (status.value,))
            rows = cursor.fetchall()
            return [self._model_from_row(row) for row in rows]
        finally:
            conn.close()

    async def list_all(self, limit: Optional[int] = None) -> List[Model]:
        """List all models."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            query = 'SELECT * FROM models ORDER BY created_at DESC'
            if limit:
                query += f' LIMIT {limit}'
            cursor.execute(query)
            rows = cursor.fetchall()
            return [self._model_from_row(row) for row in rows]
        finally:
            conn.close()

    async def update(self, model: Model) -> Model:
        """Update model."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute('''
                UPDATE models SET
                    status = %s, metadata = %s, accuracy = %s, f1_score = %s,
                    precision_score = %s, recall_score = %s, additional_metrics = %s,
                    updated_at = %s
                WHERE id = %s
                RETURNING *
            ''', (
                model.status.value,
                json.dumps(model.metadata),
                model.metrics.accuracy if model.metrics else None,
                model.metrics.f1_score if model.metrics else None,
                model.metrics.precision if model.metrics else None,
                model.metrics.recall if model.metrics else None,
                json.dumps(model.metrics.additional_metrics) if model.metrics else None,
                datetime.utcnow(),
                int(model.id)
            ))
            row = cursor.fetchone()
            conn.commit()
            return self._model_from_row(row)
        finally:
            conn.close()

    async def delete(self, model_id: str) -> bool:
        """Delete model."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM models WHERE id = %s', (int(model_id),))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted
        finally:
            conn.close()

    async def set_active(self, model_id: str, name: str) -> bool:
        """Set model as active."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # Deactivate all models with the same name
            cursor.execute('UPDATE models SET is_active = false WHERE name = %s', (name,))
            # Activate the specified model
            cursor.execute(
                'UPDATE models SET is_active = true WHERE id = %s AND name = %s',
                (int(model_id), name)
            )
            activated = cursor.rowcount > 0
            conn.commit()
            return activated
        finally:
            conn.close()