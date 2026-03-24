import logging
import time
import json
from datetime import datetime
from typing import Dict, Any
import psutil
import joblib
import os
from pathlib import Path

class ModelMonitor:
    def __init__(self, model_path: str, log_file: str = "logs/model_monitor.log"):
        self.model_path = model_path
        self.log_file = log_file

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.metrics = {
            'predictions_made': 0,
            'prediction_times': [],
            'error_count': 0,
            'model_load_time': None,
            'system_metrics': {}
        }

    def log_prediction(self, text: str, predicted_label: str, confidence: float, processing_time: float):
        self.metrics['predictions_made'] += 1
        self.metrics['prediction_times'].append(processing_time)

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'prediction',
            'text_length': len(text),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'processing_time': processing_time
        }

        self.logger.info(f"PREDICTION: {json.dumps(log_entry)}")

    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        self.metrics['error_count'] += 1

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'error_type': error_type,
            'error_message': str(error_message),
            'context': context or {}
        }

        self.logger.error(f"ERROR: {json.dumps(log_entry)}")

    def log_model_load(self, success: bool, load_time: float = None, error: str = None):
        if success:
            self.metrics['model_load_time'] = load_time
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'model_load',
                'success': True,
                'load_time': load_time,
                'model_path': self.model_path
            }
            self.logger.info(f"MODEL_LOAD: {json.dumps(log_entry)}")
        else:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'model_load',
                'success': False,
                'error': str(error),
                'model_path': self.model_path
            }
            self.logger.error(f"MODEL_LOAD_ERROR: {json.dumps(log_entry)}")

    def collect_system_metrics(self) -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }

        self.metrics['system_metrics'] = metrics
        return metrics

    def log_system_metrics(self):
        metrics = self.collect_system_metrics()
        self.logger.info(f"SYSTEM_METRICS: {json.dumps(metrics)}")

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.metrics['prediction_times']:
            return {
                'predictions_made': 0,
                'avg_prediction_time': 0,
                'error_rate': 0,
                'model_load_time': self.metrics['model_load_time']
            }

        avg_time = sum(self.metrics['prediction_times']) / len(self.metrics['prediction_times'])
        error_rate = self.metrics['error_count'] / max(self.metrics['predictions_made'], 1)

        return {
            'predictions_made': self.metrics['predictions_made'],
            'avg_prediction_time': avg_time,
            'min_prediction_time': min(self.metrics['prediction_times']),
            'max_prediction_time': max(self.metrics['prediction_times']),
            'error_rate': error_rate,
            'model_load_time': self.metrics['model_load_time'],
            'system_metrics': self.metrics['system_metrics']
        }

    def health_check(self) -> Dict[str, Any]:
        system_metrics = self.collect_system_metrics()
        performance = self.get_performance_summary()

        health_status = "healthy"
        issues = []

        if system_metrics['cpu_percent'] > 90:
            health_status = "warning"
            issues.append("High CPU usage")

        if system_metrics['memory_percent'] > 90:
            health_status = "warning"
            issues.append("High memory usage")

        if system_metrics['disk_percent'] > 90:
            health_status = "critical"
            issues.append("Low disk space")

        if performance['error_rate'] > 0.1:
            health_status = "warning"
            issues.append("High error rate")

        if not os.path.exists(self.model_path):
            health_status = "critical"
            issues.append("Model file missing")

        return {
            'status': health_status,
            'issues': issues,
            'system_metrics': system_metrics,
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        }

class DataDriftMonitor:
    def __init__(self, reference_data_path: str = None):
        self.reference_data_path = reference_data_path
        self.logger = logging.getLogger(__name__)
        self.reference_stats = self._calculate_reference_stats() if reference_data_path else None

    def _calculate_reference_stats(self) -> Dict[str, Any]:
        if not os.path.exists(self.reference_data_path):
            return None

        try:
            import pandas as pd
            df = pd.read_csv(self.reference_data_path, header=None, names=['label', 'text'])

            stats = {
                'avg_text_length': df['text'].str.len().mean(),
                'label_distribution': df['label'].value_counts().to_dict(),
                'vocab_size': len(set(' '.join(df['text']).split()))
            }

            self.logger.info(f"Reference stats calculated: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Failed to calculate reference stats: {e}")
            return None

    def check_drift(self, recent_texts: list, recent_labels: list = None) -> Dict[str, Any]:
        if not self.reference_stats:
            return {'drift_detected': False, 'reason': 'No reference data available'}

        current_avg_length = sum(len(text) for text in recent_texts) / len(recent_texts)
        length_drift = abs(current_avg_length - self.reference_stats['avg_text_length']) / self.reference_stats['avg_text_length']

        drift_threshold = 0.3  # 30% change threshold

        drift_detected = length_drift > drift_threshold

        result = {
            'drift_detected': drift_detected,
            'length_drift': length_drift,
            'current_avg_length': current_avg_length,
            'reference_avg_length': self.reference_stats['avg_text_length'],
            'threshold': drift_threshold,
            'timestamp': datetime.now().isoformat()
        }

        if drift_detected:
            self.logger.warning(f"Data drift detected: {result}")

        return result

if __name__ == "__main__":
    monitor = ModelMonitor("models/best_charge_classifier.pkl")
    health = monitor.health_check()
    print(f"Health check: {json.dumps(health, indent=2)}")

    drift_monitor = DataDriftMonitor("data/train.csv")
    sample_texts = ["Sample text for testing"]
    drift_result = drift_monitor.check_drift(sample_texts)
    print(f"Drift check: {json.dumps(drift_result, indent=2)}")