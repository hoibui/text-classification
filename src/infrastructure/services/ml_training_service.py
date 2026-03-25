import os
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

from domain.entities.model import Model, ModelMetrics, ModelStatus
from domain.services.model_training_service import ModelTrainingService
from infrastructure.config.settings import Settings


class MLTrainingService(ModelTrainingService):
    """Machine learning training service implementation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._setup_mlflow()

        # Initialize training components
        self.models = {}
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
        self.best_model_info = None

    def _setup_mlflow(self):
        """Setup MLflow configuration."""
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        mlflow.set_experiment(self.settings.mlflow.experiment_name)

    async def train_model(
        self,
        training_data_path: str,
        model_config: Dict[str, Any],
        model_name: str,
        version: str
    ) -> Model:
        """Train a new model."""
        try:
            with mlflow.start_run(run_name=self.settings.mlflow.run_name):
                # Load training data (use ALL of it)
                train_df, train_labels_encoded = self._load_data(training_data_path)
                X_train = train_df['text']
                y_train = train_labels_encoded

                # Load test data for validation
                test_data_path = training_data_path.replace('train.csv', 'test.csv')
                X_test, y_test = self._load_test_data(test_data_path)

                # Log dataset information
                mlflow.log_param("training_size", len(X_train))
                mlflow.log_param("test_size", len(X_test))
                mlflow.log_param("num_classes", len(np.unique(train_labels_encoded)))

                # Train traditional ML models
                results = self._train_traditional_models(X_train, X_test, y_train, y_test)

                # Select best model
                best_model_key = max(results.keys(), key=lambda k: results[k]['accuracy'])
                best_score = results[best_model_key]['accuracy']

                self.best_model_info = {
                    'model': self.models[best_model_key],
                    'vectorizer': self.vectorizers[best_model_key],
                    'key': best_model_key,
                    'score': best_score
                }

                # Save model
                model_path = self._save_best_model()

                # Create domain entity
                metrics = ModelMetrics(
                    accuracy=best_score,
                    f1_score=results[best_model_key]['f1_score'],
                    precision=results[best_model_key].get('precision'),
                    recall=results[best_model_key].get('recall'),
                    additional_metrics=results[best_model_key]
                )

                model = Model(
                    id=None,
                    name=model_name,
                    version=version,
                    model_type=best_model_key,
                    status=ModelStatus.READY,
                    metrics=metrics,
                    metadata=model_config,
                    file_path=model_path
                )

                self.logger.info(f"Training completed. Best model: {best_model_key} with accuracy: {best_score:.4f}")
                return model

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _load_data(self, data_path: str):
        """Load and preprocess training data."""
        self.logger.info("Loading training data...")
        df = pd.read_csv(data_path, header=None, names=['label', 'text'])

        # Remove duplicates and handle missing values
        df = df.drop_duplicates().dropna()

        # Check class distribution before filtering
        class_counts = df['label'].value_counts()
        self.logger.info(f"Original data: {len(df)} samples with {len(class_counts)} classes")
        self.logger.info(f"Class distribution:\n{class_counts}")

        # Keep ALL classes - don't filter out any data
        # Just log class distribution for information
        classes_with_few_samples = class_counts[class_counts < 2]
        if len(classes_with_few_samples) > 0:
            self.logger.info(f"Classes with only 1 sample (will use stratified split carefully):")
            for label, count in classes_with_few_samples.items():
                self.logger.info(f"  - {label}: {count} samples")

        self.logger.info(f"Using ALL data: {len(df)} samples with {len(class_counts)} classes")

        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(df['label'])

        # Final validation - ensure we have enough data for training
        if len(df) < 10:
            raise ValueError(f"Too few samples ({len(df)}). Need at least 10 samples for training.")

        unique_classes = len(df['label'].unique())
        if unique_classes < 2:
            raise ValueError(f"Too few classes ({unique_classes}). Need at least 2 classes for classification.")

        self.logger.info(f"Final dataset: {len(df)} samples with {unique_classes} classes")
        return df, labels_encoded

    def _load_test_data(self, test_data_path: str):
        """Load test data for validation."""
        self.logger.info("Loading test data...")
        test_df = pd.read_csv(test_data_path, header=None, names=['label', 'text'])

        # Remove duplicates and handle missing values
        test_df = test_df.drop_duplicates().dropna()

        # Transform test labels using the already fitted label encoder
        try:
            test_labels_encoded = self.label_encoder.transform(test_df['label'])
        except ValueError as e:
            # Handle case where test data contains labels not seen in training data
            self.logger.warning(f"Test data contains unknown labels: {e}")
            # Filter test data to only include known labels
            known_labels = set(self.label_encoder.classes_)
            test_df = test_df[test_df['label'].isin(known_labels)]
            test_labels_encoded = self.label_encoder.transform(test_df['label'])
            self.logger.info(f"Filtered test data to {len(test_df)} samples with known labels")

        self.logger.info(f"Test data loaded: {len(test_df)} samples")
        return test_df['text'], test_labels_encoded

    def _split_data(self, df, labels_encoded):
        """Split data into training and testing sets."""
        try:
            # Try stratified split first
            return train_test_split(
                df['text'],
                labels_encoded,
                test_size=self.settings.training.test_size,
                random_state=self.settings.training.random_state,
                stratify=labels_encoded
            )
        except ValueError:
            # If stratified split fails (due to classes with only 1 sample), use random split
            self.logger.warning("Stratified split failed (some classes have only 1 sample). Using random split.")
            return train_test_split(
                df['text'],
                labels_encoded,
                test_size=self.settings.training.test_size,
                random_state=self.settings.training.random_state,
                stratify=None
            )

    def _train_traditional_models(self, X_train, X_test, y_train, y_test):
        """Train traditional ML models."""
        self.logger.info("Training traditional ML models...")

        # Get model configurations
        config = self.settings.config
        vectorizer_configs = config['models']['traditional']['vectorizers']
        classifier_configs = config['models']['traditional']['classifiers']

        results = {}

        # Train each vectorizer-classifier combination
        for vec_config in vectorizer_configs:
            vectorizer = self._create_vectorizer(vec_config)

            # Fit vectorizer and transform data
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            for clf_config in classifier_configs:
                model_key = f"{vec_config['name']}_{clf_config['name']}"
                self.logger.info(f"Training {model_key}...")

                # Create and train classifier
                classifier = self._create_classifier(clf_config)
                classifier.fit(X_train_vec, y_train)

                # Evaluate model
                y_pred = classifier.predict(X_test_vec)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Perform cross-validation with error handling
                try:
                    # Check if we can do stratified cross-validation
                    unique_classes_in_train = len(set(y_train))
                    min_class_count = min([list(y_train).count(cls) for cls in set(y_train)])

                    # If any class has only 1 sample, skip cross-validation
                    if min_class_count < 2:
                        self.logger.warning(f"Skipping cross-validation for {model_key}: some classes have only 1 sample")
                        cv_scores = np.array([accuracy])
                    else:
                        cv_folds = min(self.settings.training.cv_folds, min_class_count, unique_classes_in_train)
                        cv_folds = max(2, cv_folds)  # At least 2 folds

                        cv_scores = cross_val_score(
                            classifier, X_train_vec, y_train,
                            cv=cv_folds,
                            scoring='accuracy'
                        )
                except ValueError as e:
                    self.logger.warning(f"Cross-validation failed for {model_key}: {e}")
                    # Use simple train/test split accuracy as fallback
                    cv_scores = np.array([accuracy])

                # Store results
                results[model_key] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }

                # Store model and vectorizer
                self.models[model_key] = classifier
                self.vectorizers[model_key] = vectorizer

                # Log to MLflow
                mlflow.log_metric(f"{model_key}_accuracy", accuracy)
                mlflow.log_metric(f"{model_key}_f1_score", f1)
                mlflow.log_metric(f"{model_key}_cv_mean", cv_scores.mean())

                self.logger.info(f"{model_key} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return results

    def _create_vectorizer(self, config: Dict[str, Any]):
        """Create vectorizer from configuration."""
        vectorizer_type = config['type']
        params = config.get('params', {}).copy()

        # Convert lists to tuples for parameters that need tuples
        if 'ngram_range' in params and isinstance(params['ngram_range'], list):
            params['ngram_range'] = tuple(params['ngram_range'])

        if vectorizer_type == 'TfidfVectorizer':
            return TfidfVectorizer(**params)
        elif vectorizer_type == 'CountVectorizer':
            return CountVectorizer(**params)
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")

    def _create_classifier(self, config: Dict[str, Any]):
        """Create classifier from configuration."""
        classifier_type = config['type']
        params = config.get('params', {})

        classifiers = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'SVC': SVC,
            'MultinomialNB': MultinomialNB
        }

        if classifier_type not in classifiers:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        return classifiers[classifier_type](**params)

    def _save_best_model(self):
        """Save the best model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.settings.model.save_path
        os.makedirs(model_path, exist_ok=True)

        filename = f"{model_path}/{self.settings.model.name_prefix}_{timestamp}.pkl"

        model_data = {
            'model': self.best_model_info['model'],
            'vectorizer': self.best_model_info['vectorizer'],
            'label_encoder': self.label_encoder,
            'model_key': self.best_model_info['key'],
            'score': self.best_model_info['score'],
            'config': self.settings.config
        }

        joblib.dump(model_data, filename)
        self.logger.info(f"Model saved to {filename}")

        # Log model artifact to MLflow
        mlflow.log_artifact(filename, "models")

        return filename

    async def evaluate_model(self, model: Model, test_data_path: str) -> ModelMetrics:
        """Evaluate a model on test data."""
        # Implementation for model evaluation
        # This would load the model and run it on test data
        pass

    async def compare_models(self, model_1: Model, model_2: Model) -> Dict[str, Any]:
        """Compare two models."""
        # Implementation for model comparison
        pass

    async def get_training_progress(self, training_id: str) -> Dict[str, Any]:
        """Get training progress."""
        # Implementation for getting training progress
        pass

    async def cancel_training(self, training_id: str) -> bool:
        """Cancel training."""
        # Implementation for canceling training
        pass

    async def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """Validate model configuration."""
        required_keys = ['models']
        return all(key in config for key in required_keys)