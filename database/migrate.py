#!/usr/bin/env python3
"""
Database migration script for MLOps PostgreSQL database
"""

import os
import sys
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
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

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def check_connection(self) -> bool:
        """Test database connection"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"Connected to PostgreSQL: {version}")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Connection failed: {e}")
            return False

    def create_database(self) -> bool:
        """Create database if it doesn't exist"""
        try:
            # Connect to postgres database to create mlops database
            temp_config = self.db_config.copy()
            temp_config['database'] = 'postgres'

            conn = psycopg2.connect(**temp_config)
            conn.autocommit = True
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.db_config['database'],))
            exists = cursor.fetchone()

            if not exists:
                cursor.execute(f'CREATE DATABASE "{self.db_config["database"]}"')
                logger.info(f"Database {self.db_config['database']} created")
            else:
                logger.info(f"Database {self.db_config['database']} already exists")

            conn.close()
            return True

        except psycopg2.Error as e:
            logger.error(f"Failed to create database: {e}")
            return False

    def database_exists(self) -> bool:
        """Check if the target database exists"""
        try:
            temp_config = self.db_config.copy()
            temp_config['database'] = 'postgres'

            conn = psycopg2.connect(**temp_config)
            cursor = conn.cursor()

            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.db_config['database'],))
            exists = cursor.fetchone() is not None

            conn.close()
            return exists

        except psycopg2.Error as e:
            logger.error(f"Failed to check database existence: {e}")
            return False

    def run_migration(self, migration_file: str) -> bool:
        """Run a migration SQL file"""
        try:
            with open(migration_file, 'r') as f:
                sql_content = f.read()

            conn = self._get_connection()
            cursor = conn.cursor()

            # Execute the migration
            cursor.execute(sql_content)
            conn.commit()

            logger.info(f"Migration {migration_file} completed successfully")
            conn.close()
            return True

        except FileNotFoundError:
            logger.error(f"Migration file not found: {migration_file}")
            return False
        except psycopg2.Error as e:
            logger.error(f"Migration failed: {e}")
            return False

    def get_table_info(self) -> List[Dict[str, Any]]:
        """Get information about existing tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute("""
                SELECT
                    table_name,
                    table_type,
                    table_schema
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)

            tables = cursor.fetchall()
            conn.close()

            return [dict(table) for table in tables]

        except psycopg2.Error as e:
            logger.error(f"Failed to get table info: {e}")
            return []

    def validate_schema(self) -> Dict[str, bool]:
        """Validate that all required tables exist"""
        required_tables = [
            'custom_models',
            'custom_experiments',
            'model_performance',
            'data_drift_monitoring',
            'api_usage_logs'
        ]

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            results = {}
            for table in required_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    );
                """, (table,))

                results[table] = cursor.fetchone()[0]

            conn.close()
            return results

        except psycopg2.Error as e:
            logger.error(f"Schema validation failed: {e}")
            return {}

    def backup_database(self, backup_file: str) -> bool:
        """Create a database backup using pg_dump"""
        try:
            cmd = f"""pg_dump -h {self.db_config['host']} -p {self.db_config['port']} \
                      -U {self.db_config['user']} -d {self.db_config['database']} \
                      -f {backup_file}"""

            # Set PGPASSWORD environment variable
            os.environ['PGPASSWORD'] = self.db_config['password']

            result = os.system(cmd)

            if result == 0:
                logger.info(f"Database backup created: {backup_file}")
                return True
            else:
                logger.error(f"Backup failed with exit code: {result}")
                return False

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

def main():
    migrator = DatabaseMigrator()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "init":
            logger.info("Initializing database...")

            # First ensure database exists
            if not migrator.database_exists():
                logger.info("Creating database...")
                if not migrator.create_database():
                    logger.error("Failed to create database")
                    sys.exit(1)

            # Now run migrations on the mlops database
            if migrator.run_migration("database/init.sql"):
                logger.info("Database initialization completed")

                # Validate schema
                validation = migrator.validate_schema()
                logger.info("Schema validation:")
                for table, exists in validation.items():
                    status = "✓" if exists else "✗"
                    logger.info(f"  {status} {table}")

            else:
                logger.error("Database initialization failed")
                sys.exit(1)

        elif command == "validate":
            logger.info("Validating database schema...")
            validation = migrator.validate_schema()
            all_valid = all(validation.values())

            for table, exists in validation.items():
                status = "✓" if exists else "✗"
                print(f"  {status} {table}")

            if all_valid:
                logger.info("All required tables exist")
            else:
                logger.error("Some required tables are missing")
                sys.exit(1)

        elif command == "info":
            logger.info("Database table information:")
            tables = migrator.get_table_info()
            for table in tables:
                print(f"  - {table['table_name']} ({table['table_type']})")

        elif command == "backup":
            backup_file = sys.argv[2] if len(sys.argv) > 2 else "backup.sql"
            migrator.backup_database(backup_file)

        else:
            logger.error(f"Unknown command: {command}")
            sys.exit(1)

    else:
        logger.info("Available commands:")
        logger.info("  init     - Initialize database and run migrations")
        logger.info("  validate - Validate database schema")
        logger.info("  info     - Show table information")
        logger.info("  backup   - Create database backup")

if __name__ == "__main__":
    main()