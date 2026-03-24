#!/usr/bin/env python3
"""
Setup script for MLflow tracking database
"""

import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mlflow_database():
    """Create a separate database for MLflow tracking"""
    try:
        # Connect to postgres database to create mlflow_tracking database
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="hoibui",
            password="admin"
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'mlflow_tracking'")
        exists = cursor.fetchone()

        if not exists:
            cursor.execute('CREATE DATABASE "mlflow_tracking"')
            logger.info("MLflow tracking database created")
        else:
            logger.info("MLflow tracking database already exists")

        conn.close()
        return True

    except psycopg2.Error as e:
        logger.error(f"Failed to create MLflow database: {e}")
        return False

def main():
    logger.info("Setting up MLflow tracking database...")

    if create_mlflow_database():
        logger.info("✅ MLflow database setup complete!")
        logger.info("You can now run the trainer: python src/trainer.py")
    else:
        logger.error("❌ MLflow database setup failed")

if __name__ == "__main__":
    main()