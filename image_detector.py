from PIL import Image
import numpy as np
import os
import pickle


IMAGE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "image_model.pkl")

_image_model = None


def _find_class_index(classes, target_label):
    target_upper = str(target_label).upper()
    for index, value in enumerate(classes):
        if str(value).upper() == target_upper:
            return index
    return None


def extract_image_features(image):
    width, height = image.size
    resolution = width * height
    aspect_ratio = width / height if height else 1

    rgb_image = image.convert("RGB")
    img_array = np.array(rgb_image)
    variance = float(np.var(img_array))
    exif_data = image.getexif()

    # Approximate compression artifacts by measuring abrupt pixel changes.
    horizontal_diff = np.mean(np.abs(np.diff(img_array.astype(np.float32), axis=1)))
    vertical_diff = np.mean(np.abs(np.diff(img_array.astype(np.float32), axis=0)))
    artifact_score = (horizontal_diff + vertical_diff) / 2
    metadata_present = 1.0 if exif_data and len(exif_data) > 0 else 0.0

    return np.array(
        [
            float(width),
            float(height),
            float(resolution),
            float(aspect_ratio),
            float(variance),
            float(artifact_score),
            metadata_present,
        ],
        dtype=float,
    )


def load_image_model():
    global _image_model

    if _image_model is not None:
        return _image_model

    if not os.path.exists(IMAGE_MODEL_PATH):
        return None

    with open(IMAGE_MODEL_PATH, "rb") as model_file:
        _image_model = pickle.load(model_file)

    return _image_model


def analyze_image_score(image):
    features = extract_image_features(image)
    width, height, resolution, aspect_ratio, variance, artifact_score, metadata_present = features

    suspicious_score = 0.0
    reasons = []

    if resolution < 150000:
        suspicious_score += 0.18
        reasons.append("Low resolution reduces reliability")
    elif resolution > 12000000:
        suspicious_score += 0.14
        reasons.append("Extremely high resolution looks unusual")
    else:
        reasons.append("Resolution looks reasonable")

    if aspect_ratio < 0.55 or aspect_ratio > 2.1:
        suspicious_score += 0.12
        reasons.append("Unusual aspect ratio was detected")
    else:
        reasons.append("Aspect ratio looks natural")

    if variance < 300:
        suspicious_score += 0.24
        reasons.append("Low visual variance suggests synthetic or over-smoothed content")
    elif variance < 800:
        suspicious_score += 0.12
        reasons.append("Moderate noise pattern gives a slight warning")
    else:
        reasons.append("Noise and variance look natural")

    if image.format not in ["JPEG", "PNG"]:
        suspicious_score += 0.10
        reasons.append("Unusual image format")

    if metadata_present:
        reasons.append("EXIF metadata is present")
    else:
        suspicious_score += 0.10
        reasons.append("Missing EXIF metadata lowers trust")

    if artifact_score > 45:
        suspicious_score += 0.14
        reasons.append("Compression artifacts appear stronger than normal")
    elif artifact_score > 25:
        suspicious_score += 0.08
        reasons.append("Mild compression artifacts are visible")
    else:
        reasons.append("Compression pattern looks normal")

    score = max(0.0, min(round(suspicious_score, 2), 1.0))
    explanation = ", ".join(reasons)
    return score, explanation


def detect_image(image_path):
    try:
        image = Image.open(image_path)
        image_model = load_image_model()

        if image_model is not None:
            features = extract_image_features(image).reshape(1, -1)
            probabilities = image_model.predict_proba(features)[0]
            classes = list(image_model.classes_)
            real_index = _find_class_index(classes, "REAL")
            fake_index = _find_class_index(classes, "FAKE")

            if real_index is not None:
                real_score = float(probabilities[real_index])
            elif fake_index is not None:
                real_score = 1.0 - float(probabilities[fake_index])
            else:
                real_score = 0.5

            if real_score >= 0.5:
                label = "REAL"
                confidence = round(real_score * 100, 2)
            else:
                label = "FAKE"
                confidence = round((1 - real_score) * 100, 2)

            score, heuristic_explanation = analyze_image_score(image)
            explanation = (
                "Image prediction used the trained visual model from your dataset. "
                + heuristic_explanation
            )
            return label, confidence, explanation

        score, explanation = analyze_image_score(image)

        if score >= 0.5:
            label = "FAKE"
            confidence = min(95, round(50 + (score * 45), 2))
        else:
            label = "REAL"
            confidence = min(95, round(55 + ((1 - score) * 40), 2))

        return label, confidence, explanation

    except Exception as e:
        return "ERROR", 50, str(e)
