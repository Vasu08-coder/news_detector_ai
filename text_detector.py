import os
import pickle
import re
import string

from sklearn.feature_extraction import text as sklearn_text


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
STOP_WORDS = set(sklearn_text.ENGLISH_STOP_WORDS)

_model = None
_vectorizer = None


def load_model_files():
    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("model.pkl or vectorizer.pkl not found.")

    with open(MODEL_PATH, "rb") as model_file:
        _model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        _vectorizer = pickle.load(vectorizer_file)

    return _model, _vectorizer


def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(words)


def detect_text(text):
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            return {
                "label": "UNKNOWN",
                "confidence": 0,
                "explanation": "Model not trained. Please run train_model.py first.",
            }

        model, vectorizer = load_model_files()
        cleaned_text = preprocess_text(text)

        text_vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]

        classes = list(model.classes_)
        predicted_index = classes.index(prediction)
        raw_confidence = float(probabilities[predicted_index])
        label = str(prediction).upper()
        confidence_percent = round(raw_confidence * 100, 2)

        if len(cleaned_text.split()) < 5:
            quality_note = "The input text is short, so the prediction is more uncertain."
        else:
            quality_note = "The prediction is based on a trained NLP model analyzing text patterns and word usage."

        explanation = f"{quality_note} The trained text model classified this input as {label}."

        print("ML prediction used")

        return {
            "label": label,
            "confidence": confidence_percent,
            "explanation": explanation,
        }
    except Exception as error:
        return {
            "label": "UNKNOWN",
            "confidence": 0,
            "explanation": f"Text detection failed: {error}",
        }
