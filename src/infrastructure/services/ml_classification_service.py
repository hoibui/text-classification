import asyncio
import logging
import joblib
import numpy as np
from typing import List, Dict
from datetime import datetime

from domain.entities.text_classification import TextClassification, ClassificationResult
from domain.services.classification_service import ClassificationService


class MLClassificationService(ClassificationService):
    """ML-based classification service using trained models."""

    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path or "models/best_model_20260325_091723.pkl"
        self.model_data = None
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.model_key = None
        # Extract filename as version
        import os
        self.model_version = os.path.basename(self.model_path).replace('.pkl', '')
        self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            self.model_data = joblib.load(self.model_path)

            self.model = self.model_data['model']
            self.vectorizer = self.model_data['vectorizer']
            self.label_encoder = self.model_data['label_encoder']
            self.model_key = self.model_data['model_key']

            self.logger.info(f"Model loaded successfully: {self.model_key}")
            self.logger.info(f"Available labels: {list(self.label_encoder.classes_)}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    async def classify_text(self, text_classification: TextClassification) -> ClassificationResult:
        """Classify a single text and return the result."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        try:
            start_time = datetime.utcnow()

            # Vectorize the text
            text_vectorized = self.vectorizer.transform([text_classification.text])

            # Predict
            prediction = self.model.predict(text_vectorized)[0]
            probabilities = self.model.predict_proba(text_vectorized)[0]

            # Debug: Log raw probabilities to see actual precision
            self.logger.info(f"Raw probabilities: {probabilities[:5]}")  # Log first 5 values

            # Convert prediction back to original label
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]

            # Get confidence (max probability) - keep full precision
            confidence = float(max(probabilities))

            # Create probabilities dict with original labels - keep full precision
            prob_dict = {}
            for i, prob in enumerate(probabilities):
                original_label = self.label_encoder.inverse_transform([i])[0]
                prob_dict[original_label] = float(prob)

            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            return ClassificationResult(
                text=text_classification.text,
                predicted_label=predicted_label,
                confidence=confidence,
                probabilities=prob_dict,
                model_version=f"{self.model_key}-{self.model_version}",
                processing_time=processing_time,
                request_id=text_classification.request_id,
                timestamp=end_time
            )

        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            raise RuntimeError(f"Classification failed: {str(e)}")

    async def classify_batch(self, text_classifications: List[TextClassification]) -> List[ClassificationResult]:
        """Classify multiple texts and return the results."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        results = []
        try:
            # Extract texts for batch processing
            texts = [tc.text for tc in text_classifications]

            start_time = datetime.utcnow()

            # Vectorize all texts
            texts_vectorized = self.vectorizer.transform(texts)

            # Predict all at once
            predictions = self.model.predict(texts_vectorized)
            probabilities = self.model.predict_proba(texts_vectorized)

            end_time = datetime.utcnow()
            total_processing_time = (end_time - start_time).total_seconds()
            per_text_time = total_processing_time / len(texts)

            # Process results
            for i, text_classification in enumerate(text_classifications):
                # Convert prediction back to original label
                predicted_label = self.label_encoder.inverse_transform([predictions[i]])[0]

                # Get confidence (max probability) formatted to exactly 4 decimal places
                confidence = float(f"{float(max(probabilities[i])):.4f}")

                # Create probabilities dict with original labels (exactly 4 decimal places)
                prob_dict = {}
                for j, prob in enumerate(probabilities[i]):
                    original_label = self.label_encoder.inverse_transform([j])[0]
                    prob_dict[original_label] = float(f"{float(prob):.4f}")

                result = ClassificationResult(
                    text=text_classification.text,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    probabilities=prob_dict,
                    model_version=f"{self.model_key}-{self.model_version}",
                    processing_time=per_text_time,
                    request_id=text_classification.request_id,
                    timestamp=end_time
                )
                results.append(result)

        except Exception as e:
            self.logger.error(f"Batch classification failed: {str(e)}")
            raise RuntimeError(f"Batch classification failed: {str(e)}")

        return results

    async def get_model_info(self) -> Dict[str, any]:
        """Get information about the currently loaded model."""
        if not self.model:
            return {"status": "not_loaded"}

        return {
            "model_name": self.model_key,
            "model_version": self.model_version,
            "model_type": "ml_trained",
            "status": "ready",
            "loaded_at": datetime.utcnow().isoformat(),
            "supported_labels": list(self.label_encoder.classes_),
            "model_path": self.model_path
        }

    async def is_model_ready(self) -> bool:
        """Check if the classification model is ready for inference."""
        return self.model is not None

    async def reload_model(self, model_path: str = None) -> bool:
        """Reload the classification model. Returns True if successful."""
        try:
            if model_path:
                self.model_path = model_path
            self._load_model()
            return True
        except Exception as e:
            self.logger.error(f"Model reload failed: {str(e)}")
            return False