import joblib
import argparse
import sys

def load_model(model_path='best_charge_classifier.pkl'):
    try:
        model_data = joblib.load(model_path)
        return model_data
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_single(text, model_data):
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    label_encoder = model_data['label_encoder']

    text_cleaned = str(text).lower().strip()
    X_vec = vectorizer.transform([text_cleaned])
    prediction = model.predict(X_vec)[0]
    probability = model.predict_proba(X_vec)[0]

    predicted_label = label_encoder.inverse_transform([prediction])[0]
    confidence = max(probability)

    return predicted_label, confidence

def main():
    parser = argparse.ArgumentParser(description='Predict charge type for given text')
    parser.add_argument('--text', required=True, help='Text to classify')
    parser.add_argument('--model', default='best_charge_classifier.pkl', help='Path to saved model')

    args = parser.parse_args()

    model_data = load_model(args.model)
    predicted_label, confidence = predict_single(args.text, model_data)

    print(f"Text: '{args.text}'")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.3f}")

if __name__ == "__main__":
    main()