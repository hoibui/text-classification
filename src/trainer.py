import os
import yaml
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
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

class ModelTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.setup_logging()
        self.setup_mlflow()

        self.models = {}
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
        self.best_model_info = None

    def setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        logging_config = self.config['logging']
        logging.basicConfig(
            level=getattr(logging, logging_config['root']['level']),
            format=logging_config['formatters']['standard']['format']
        )
        self.logger = logging.getLogger(__name__)

    def setup_mlflow(self):
        if self.config['mlops']['experiment_tracking']['enabled']:
            mlflow.set_tracking_uri(self.config['mlops']['experiment_tracking']['tracking_uri'])
            mlflow.set_experiment("charge_classification")

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        self.logger.info("Loading training data...")
        df = pd.read_csv(self.config['training']['data_path'], header=None, names=['label', 'text'])

        self.logger.info(f"Dataset shape: {df.shape}")
        self.logger.info(f"Number of unique labels: {df['label'].nunique()}")

        df['text'] = df['text'].fillna('').astype(str)
        df['text_cleaned'] = df['text'].str.lower().str.strip()

        labels_encoded = self.label_encoder.fit_transform(df['label'])

        return df, labels_encoded

    def split_data(self, df: pd.DataFrame, labels: np.ndarray) -> Tuple:
        # Check if stratified split is possible
        unique, counts = np.unique(labels, return_counts=True)
        min_class_count = np.min(counts)

        if min_class_count < 2:
            self.logger.warning(f"Some classes have only {min_class_count} sample(s). Using non-stratified split.")
            return train_test_split(
                df['text_cleaned'],
                labels,
                test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state'],
                stratify=None
            )
        else:
            return train_test_split(
                df['text_cleaned'],
                labels,
                test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state'],
                stratify=labels
            )

    def get_vectorizer(self, vec_config: Dict[str, Any]):
        # Convert list parameters to tuples where needed
        params = vec_config['params'].copy()
        if 'ngram_range' in params:
            params['ngram_range'] = tuple(params['ngram_range'])

        if vec_config['type'] == 'TfidfVectorizer':
            return TfidfVectorizer(**params)
        elif vec_config['type'] == 'CountVectorizer':
            return CountVectorizer(**params)
        else:
            raise ValueError(f"Unknown vectorizer type: {vec_config['type']}")

    def get_classifier(self, clf_config: Dict[str, Any]):
        classifiers = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'SVC': SVC,
            'MultinomialNB': MultinomialNB
        }

        clf_class = classifiers[clf_config['type']]
        return clf_class(**clf_config['params'])

    def train_traditional_models(self, X_train, X_test, y_train, y_test) -> Dict[str, float]:
        self.logger.info("Training traditional ML models...")

        results = {}
        best_score = 0
        best_model_key = None

        for vec_config in self.config['models']['traditional']['vectorizers']:
            vec_name = vec_config['name']
            vectorizer = self.get_vectorizer(vec_config)

            self.logger.info(f"Processing with {vec_name} vectorizer...")
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            for clf_config in self.config['models']['traditional']['classifiers']:
                clf_name = clf_config['name']
                classifier = self.get_classifier(clf_config)

                model_key = f"{vec_name}_{clf_name}"
                self.logger.info(f"Training {model_key}...")

                with mlflow.start_run(run_name=model_key, nested=True):
                    classifier.fit(X_train_vec, y_train)
                    y_pred = classifier.predict(X_test_vec)

                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    mlflow.log_param("vectorizer", vec_name)
                    mlflow.log_param("classifier", clf_name)
                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("f1_score", f1)

                    mlflow.sklearn.log_model(classifier, "model")

                    self.models[model_key] = classifier
                    self.vectorizers[model_key] = vectorizer
                    results[model_key] = accuracy

                    self.logger.info(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

                    if accuracy > best_score:
                        best_score = accuracy
                        best_model_key = model_key

        self.best_model_info = {
            'key': best_model_key,
            'score': best_score,
            'model': self.models[best_model_key],
            'vectorizer': self.vectorizers[best_model_key]
        }

        self.logger.info(f"Best traditional model: {best_model_key} with accuracy: {best_score:.4f}")
        return results

    def save_best_model(self, filename: str = None):
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"models/best_model_{timestamp}.pkl"

        os.makedirs("models", exist_ok=True)

        model_data = {
            'model': self.best_model_info['model'],
            'vectorizer': self.best_model_info['vectorizer'],
            'label_encoder': self.label_encoder,
            'model_key': self.best_model_info['key'],
            'score': self.best_model_info['score'],
            'config': self.config
        }

        joblib.dump(model_data, filename)
        self.logger.info(f"Best model saved to {filename}")

        if self.config['mlops']['experiment_tracking']['enabled']:
            mlflow.log_artifact(filename, "models")

        return filename

    def train(self) -> str:
        with mlflow.start_run(run_name="charge_classification_training"):
            df, labels_encoded = self.load_data()
            X_train, X_test, y_train, y_test = self.split_data(df, labels_encoded)

            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("num_classes", len(np.unique(labels_encoded)))
            mlflow.log_param("test_size", self.config['training']['test_size'])

            traditional_results = self.train_traditional_models(X_train, X_test, y_train, y_test)

            mlflow.log_metric("best_accuracy", self.best_model_info['score'])
            mlflow.log_param("best_model", self.best_model_info['key'])

            model_path = self.save_best_model()

            self.logger.info("Training completed successfully!")
            return model_path

if __name__ == "__main__":
    trainer = ModelTrainer()
    model_path = trainer.train()
    print(f"Training completed. Best model saved at: {model_path}")