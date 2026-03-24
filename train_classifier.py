import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class ChargeClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class ChargeTypeClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {}
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_score = 0

    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.data_path, header=None, names=['label', 'text'])

        print(f"Dataset shape: {df.shape}")
        print(f"Number of unique labels: {df['label'].nunique()}")
        print(f"Label distribution:\n{df['label'].value_counts()}")

        df['text'] = df['text'].fillna('').astype(str)
        df['text_length'] = df['text'].str.len()
        df['text_cleaned'] = df['text'].str.lower().str.strip()

        self.df = df
        self.labels_encoded = self.label_encoder.fit_transform(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(
            df['text_cleaned'],
            self.labels_encoded,
            test_size=0.2,
            random_state=42,
            stratify=self.labels_encoded
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        return X_train, X_test, y_train, y_test

    def train_traditional_models(self):
        print("\nTraining traditional ML models...")

        vectorizers = {
            'tfidf_word': TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english'),
            'tfidf_char': TfidfVectorizer(max_features=10000, analyzer='char', ngram_range=(2, 4)),
            'count': CountVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        }

        classifiers = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }

        best_combo = None
        best_score = 0

        for vec_name, vectorizer in vectorizers.items():
            print(f"\nTesting with {vec_name} vectorizer...")
            X_train_vec = vectorizer.fit_transform(self.X_train)
            X_test_vec = vectorizer.transform(self.X_test)

            for clf_name, classifier in classifiers.items():
                print(f"  Training {clf_name}...")

                classifier.fit(X_train_vec, self.y_train)
                y_pred = classifier.predict(X_test_vec)
                accuracy = accuracy_score(self.y_test, y_pred)

                model_key = f"{vec_name}_{clf_name}"
                self.models[model_key] = classifier
                self.vectorizers[model_key] = vectorizer

                print(f"    Accuracy: {accuracy:.4f}")

                if accuracy > best_score:
                    best_score = accuracy
                    best_combo = model_key
                    self.best_model = classifier
                    self.best_vectorizer = vectorizer

        print(f"\nBest traditional model: {best_combo} with accuracy: {best_score:.4f}")
        return best_combo, best_score

    def train_transformer_model(self):
        print("\nTraining transformer model...")

        try:
            model_name = "distilbert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(np.unique(self.labels_encoded))
            )

            train_dataset = ChargeClassificationDataset(
                self.X_train.tolist(),
                self.y_train,
                tokenizer
            )
            test_dataset = ChargeClassificationDataset(
                self.X_test.tolist(),
                self.y_test,
                tokenizer
            )

            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                data_collator=DataCollatorWithPadding(tokenizer),
            )

            trainer.train()

            eval_results = trainer.evaluate()
            transformer_accuracy = 1 - eval_results['eval_loss']  # Approximate accuracy

            self.models['transformer'] = (model, tokenizer, trainer)
            print(f"Transformer model evaluation loss: {eval_results['eval_loss']:.4f}")

            return transformer_accuracy

        except Exception as e:
            print(f"Transformer training failed: {e}")
            print("Continuing with traditional models only...")
            return 0

    def evaluate_best_model(self):
        print(f"\nEvaluating best model...")

        if hasattr(self, 'best_vectorizer'):
            X_test_vec = self.best_vectorizer.transform(self.X_test)
            y_pred = self.best_model.predict(X_test_vec)
        else:
            print("No best model found!")
            return

        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")

        print(f"\nClassification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_pred, target_names=target_names))

        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved as 'confusion_matrix.png'")

    def save_model(self, filename='best_charge_classifier.pkl'):
        print(f"\nSaving best model to {filename}...")
        model_data = {
            'model': self.best_model,
            'vectorizer': self.best_vectorizer,
            'label_encoder': self.label_encoder,
            'model_type': 'traditional'
        }
        joblib.dump(model_data, filename)
        print(f"Model saved successfully!")

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if hasattr(self, 'best_vectorizer'):
            texts_cleaned = [str(text).lower().strip() for text in texts]
            X_vec = self.best_vectorizer.transform(texts_cleaned)
            predictions = self.best_model.predict(X_vec)
            probabilities = self.best_model.predict_proba(X_vec)

            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                predicted_label = self.label_encoder.inverse_transform([pred])[0]
                confidence = np.max(prob)
                results.append({
                    'text': texts[i],
                    'predicted_label': predicted_label,
                    'confidence': confidence
                })

            return results
        else:
            print("No trained model available!")
            return None

def main():
    print("Starting Charge Type Classification Training...")

    classifier = ChargeTypeClassifier('data/train.csv')

    classifier.load_and_preprocess_data()

    traditional_combo, traditional_score = classifier.train_traditional_models()

    transformer_score = classifier.train_transformer_model()

    if transformer_score > traditional_score:
        print(f"\nTransformer model performed better: {transformer_score:.4f}")
    else:
        print(f"\nTraditional model performed better: {traditional_score:.4f}")
        classifier.evaluate_best_model()
        classifier.save_model()

    print("\nTesting predictions with sample texts...")
    sample_texts = [
        "Peak energy usage charge",
        "Network connection fee",
        "Solar generation credit",
        "Metering service charge"
    ]

    predictions = classifier.predict(sample_texts)
    if predictions:
        for pred in predictions:
            print(f"Text: '{pred['text']}' -> {pred['predicted_label']} (confidence: {pred['confidence']:.3f})")

if __name__ == "__main__":
    main()