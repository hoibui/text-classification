-- Initialize MLOps database with required tables and users

-- Create database if it doesn't exist (handled by Docker/environment)
-- CREATE DATABASE IF NOT EXISTS mlops;

-- Note: Database connection should already be to 'mlops' database

-- MLflow will create its own tables (experiments, runs, etc.) automatically
-- We create our custom tables with different names to avoid conflicts

-- Create custom model registry table (separate from MLflow's model registry)
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
);

-- Create custom experiments table (separate from MLflow's experiments)
CREATE TABLE IF NOT EXISTS custom_experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    model_id INTEGER REFERENCES custom_models(id),
    parameters JSONB,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_custom_models_name ON custom_models(name);
CREATE INDEX IF NOT EXISTS idx_custom_models_active ON custom_models(name, is_active);
CREATE INDEX IF NOT EXISTS idx_custom_models_created_at ON custom_models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_custom_experiments_model ON custom_experiments(model_id);
CREATE INDEX IF NOT EXISTS idx_custom_experiments_created_at ON custom_experiments(created_at DESC);

-- Create performance monitoring table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES custom_models(id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_version VARCHAR(50)
);

CREATE INDEX IF NOT EXISTS idx_performance_model_date ON model_performance(model_id, measurement_date DESC);

-- Create data drift monitoring table
CREATE TABLE IF NOT EXISTS data_drift_monitoring (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES custom_models(id),
    feature_name VARCHAR(255),
    drift_score FLOAT,
    threshold_exceeded BOOLEAN DEFAULT FALSE,
    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reference_period_start TIMESTAMP,
    reference_period_end TIMESTAMP,
    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_drift_model_date ON data_drift_monitoring(model_id, check_date DESC);

-- Create API usage logs table
CREATE TABLE IF NOT EXISTS api_usage_logs (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms FLOAT,
    prediction_confidence FLOAT,
    predicted_label VARCHAR(255),
    input_text_hash VARCHAR(32),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_agent TEXT,
    ip_address INET
);

CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON api_usage_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_usage_endpoint ON api_usage_logs(endpoint, timestamp DESC);

-- Insert initial data if needed
INSERT INTO custom_models (name, version, file_path, metadata, performance_metrics, is_active, model_hash)
VALUES ('charge_classifier', '1.0.0', 'models/initial_model.pkl',
        '{"description": "Initial charge classification model", "training_date": "2024-01-01"}',
        '{"accuracy": 0.85, "f1_score": 0.83}',
        false, 'placeholder_hash')
ON CONFLICT (name, version) DO NOTHING;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hoibui;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hoibui;

-- Create view for model performance summary
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT
    m.id,
    m.name,
    m.version,
    m.is_active,
    m.created_at,
    m.performance_metrics,
    COUNT(e.id) as experiment_count,
    MAX(e.created_at) as last_experiment_date
FROM custom_models m
LEFT JOIN custom_experiments e ON m.id = e.model_id
GROUP BY m.id, m.name, m.version, m.is_active, m.created_at, m.performance_metrics
ORDER BY m.created_at DESC;

-- Create view for recent API usage statistics
CREATE OR REPLACE VIEW api_usage_summary AS
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    endpoint,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    AVG(prediction_confidence) as avg_confidence,
    COUNT(*) FILTER (WHERE status_code >= 400) as error_count
FROM api_usage_logs
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp), endpoint
ORDER BY hour DESC, request_count DESC;

COMMENT ON TABLE custom_models IS 'Registry of trained models with versioning';
COMMENT ON TABLE custom_experiments IS 'Experiment tracking for model development';
COMMENT ON TABLE model_performance IS 'Historical performance metrics for models';
COMMENT ON TABLE data_drift_monitoring IS 'Data drift detection results';
COMMENT ON TABLE api_usage_logs IS 'API request logs for monitoring and analytics';