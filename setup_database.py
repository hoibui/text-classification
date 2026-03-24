#!/usr/bin/env python3
"""
Quick setup script for PostgreSQL database initialization
"""

import sys
import subprocess
from database.migrate import DatabaseMigrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Setting up MLOps database...")

    migrator = DatabaseMigrator()

    # First ensure database exists
    if not migrator.database_exists():
        logger.info("Creating database...")
        if not migrator.create_database():
            logger.error("Failed to create database. Please ensure:")
            logger.error("  - PostgreSQL is running on localhost:5432")
            logger.error("  - User 'hoibui' has superuser privileges or CREATE DATABASE permission")
            sys.exit(1)

    # Test connection to mlops database
    if not migrator.check_connection():
        logger.error("Cannot connect to mlops database. Please check configuration.")
        sys.exit(1)

    # Run initialization
    logger.info("Running database migrations...")
    if migrator.run_migration("database/init.sql"):
        logger.info("Database setup completed successfully!")

        # Validate schema
        validation = migrator.validate_schema()
        logger.info("Schema validation:")
        for table, exists in validation.items():
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {table}")

        if all(validation.values()):
            logger.info("✅ Database setup complete and validated!")
        else:
            logger.warning("⚠️  Some tables may be missing")

    else:
        logger.error("❌ Database setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()