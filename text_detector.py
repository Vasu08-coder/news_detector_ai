import os
import pickle
import re
import string

from sklearn.feature_extraction import text as sklearn_text


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
STOP_WORDS = set(sklearn_text.ENGLISH_STOP_WORDS)
COMMON_NEWS_KEYWORDS = {
    "government", "president", "minister", "election", "court", "police",
    "report", "official", "statement", "international", "india", "world",
    "economy", "health", "education", "technology", "market", "sports",
    "policy", "agency", "minister", "parliament", "cabinet",
}
SUSPICIOUS_KEYWORDS = {
    "shocking", "viral", "rumor", "secret", "conspiracy", "aliens",
    "miracle", "exposed", "banned", "click", "unbelievable", "hoax",
}

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
        real_index = None
        for index, class_name in enumerate(classes):
            if str(class_name).strip().upper() in {"1", "REAL", "TRUE"}:
                real_index = index
                break

        if real_index is not None:
            model_real_probability = float(probabilities[real_index])
        else:
            normalized_prediction = str(prediction).strip().upper()
            model_real_probability = raw_confidence if normalized_prediction in {"1", "REAL", "TRUE"} else (1.0 - raw_confidence)

        normalized_prediction = str(prediction).strip().upper()
        if normalized_prediction in {"1", "REAL", "TRUE"}:
            label = "REAL"
        elif normalized_prediction in {"0", "FAKE", "FALSE"}:
            label = "FAKE"
        else:
            label = normalized_prediction

        tokens = cleaned_text.split()
        common_matches = sorted({word for word in tokens if word in COMMON_NEWS_KEYWORDS})
        suspicious_matches = sorted({word for word in tokens if word in SUSPICIOUS_KEYWORDS})

        heuristic_real_probability = 0.5
        if len(common_matches) >= 2 and not suspicious_matches:
            heuristic_real_probability = 0.78
        elif common_matches and not suspicious_matches:
            heuristic_real_probability = 0.65
        elif len(suspicious_matches) >= 2 and not common_matches:
            heuristic_real_probability = 0.18
        elif suspicious_matches:
            heuristic_real_probability = 0.32

        blended_real_probability = (0.55 * model_real_probability) + (0.45 * heuristic_real_probability)

        if len(common_matches) >= 2 and not suspicious_matches and blended_real_probability < 0.5:
            blended_real_probability = min(0.68, blended_real_probability + 0.15)
        if len(suspicious_matches) >= 2 and blended_real_probability > 0.5:
            blended_real_probability = max(0.32, blended_real_probability - 0.18)

        blended_real_probability = max(0.0, min(blended_real_probability, 1.0))
        if blended_real_probability >= 0.5:
            label = "REAL"
            confidence_percent = round(blended_real_probability * 100, 2)
        else:
            label = "FAKE"
            confidence_percent = round((1.0 - blended_real_probability) * 100, 2)

        if len(cleaned_text.split()) < 5:
            quality_note = "The input text is short, so the prediction is more uncertain."
        else:
            quality_note = "The prediction is based on a trained NLP model plus a simple credibility check on the wording."

        detail_parts = []
        if common_matches:
            detail_parts.append(f"credible news terms found: {', '.join(common_matches[:4])}")
        if suspicious_matches:
            detail_parts.append(f"suspicious terms found: {', '.join(suspicious_matches[:4])}")

        if detail_parts:
            detail_text = " " + " ".join(detail_parts) + "."
        else:
            detail_text = ""

        explanation = f"{quality_note} The text detector classified this input as {label}.{detail_text}"

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
