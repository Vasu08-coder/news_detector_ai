import os
import pickle

import cv2
import numpy as np


VIDEO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "video_model.pkl")

_video_model = None


def _find_class_index(classes, target_label):
    target_upper = str(target_label).upper()
    for index, value in enumerate(classes):
        if str(value).upper() == target_upper:
            return index
    return None


def load_video_model():
    global _video_model

    if _video_model is not None:
        return _video_model

    if not os.path.exists(VIDEO_MODEL_PATH):
        return None

    with open(VIDEO_MODEL_PATH, "rb") as model_file:
        _video_model = pickle.load(model_file)

    return _video_model


def extract_video_features(video_path):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        return None, "Could not open the uploaded video."

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)

    if total_frames <= 0:
        video.release()
        return None, "No readable frames were found in the video."

    sample_count = min(5, total_frames)
    if total_frames <= sample_count:
        sample_indices = list(range(total_frames))
    else:
        sample_indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()

    blur_values = []
    brightness_values = []
    frame_differences = []
    previous_gray = None
    checked_frames = 0

    for frame_index in sample_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video.read()
        if not success:
            continue

        checked_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness_values.append(float(gray.mean()))

        if previous_gray is not None:
            resized_previous = cv2.resize(previous_gray, (gray.shape[1], gray.shape[0]))
            frame_differences.append(float(np.mean(cv2.absdiff(gray, resized_previous))))

        previous_gray = gray

    video.release()

    if checked_frames == 0:
        return None, "No readable frames were found in the video."

    blur_mean = float(np.mean(blur_values)) if blur_values else 0.0
    blur_std = float(np.std(blur_values)) if blur_values else 0.0
    brightness_mean = float(np.mean(brightness_values)) if brightness_values else 0.0
    brightness_range = float(max(brightness_values) - min(brightness_values)) if len(brightness_values) > 1 else 0.0
    frame_diff_mean = float(np.mean(frame_differences)) if frame_differences else 0.0

    features = np.array(
        [
            float(total_frames),
            float(fps),
            float(checked_frames),
            blur_mean,
            blur_std,
            brightness_mean,
            brightness_range,
            frame_diff_mean,
        ],
        dtype=float,
    )

    return features, {
        "checked_frames": checked_frames,
        "blur_mean": blur_mean,
        "blur_std": blur_std,
        "brightness_range": brightness_range,
        "frame_diff_mean": frame_diff_mean,
    }


def analyze_video_score(video_path):
    extracted = extract_video_features(video_path)
    if extracted[0] is None:
        return None, extracted[1]

    _, details = extracted
    checked_frames = details["checked_frames"]
    blur_mean = details["blur_mean"]
    brightness_range = details["brightness_range"]
    frame_diff_mean = details["frame_diff_mean"]

    suspicious_score = 0.0
    reasons = [f"{checked_frames} frames were analyzed from the video"]

    if blur_mean < 80:
        suspicious_score += 0.28
        reasons.append("Several frames looked blurry")
    elif blur_mean < 140:
        suspicious_score += 0.12
        reasons.append("A few frames showed noticeable blur")
    else:
        reasons.append("Frames looked reasonably sharp")

    if brightness_range < 8:
        suspicious_score += 0.22
        reasons.append("Brightness stayed unusually uniform across frames")
    elif brightness_range < 18:
        suspicious_score += 0.1
        reasons.append("Brightness changes were slightly limited")
    else:
        reasons.append("Brightness changed naturally across frames")

    if frame_diff_mean < 6:
        suspicious_score += 0.14
        reasons.append("Frame-to-frame variation looked limited")
    else:
        reasons.append("Frame variation looked natural")

    score = max(0.0, min(round(suspicious_score, 2), 1.0))
    explanation = ", ".join(reasons)
    return score, explanation


def detect_video(video_path):
    try:
        video_model = load_video_model()

        if video_model is not None:
            extracted = extract_video_features(video_path)
            if extracted[0] is None:
                return "ERROR", 50, extracted[1]

            features, details = extracted
            probabilities = video_model.predict_proba(features.reshape(1, -1))[0]
            classes = list(video_model.classes_)
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

            explanation = (
                "Video prediction used the trained video model. "
                f"{details['checked_frames']} frames were analyzed, "
                f"average blur was {details['blur_mean']:.2f}, "
                f"brightness range was {details['brightness_range']:.2f}, "
                f"and frame variation was {details['frame_diff_mean']:.2f}."
            )
            return label, confidence, explanation

        score, explanation = analyze_video_score(video_path)

        if score is None:
            return "ERROR", 50, explanation

        if score >= 0.5:
            label = "FAKE"
            confidence = min(95, round(50 + (score * 45), 2))
        else:
            label = "REAL"
            confidence = min(95, round(55 + ((1 - score) * 40), 2))

        return label, confidence, explanation
    except Exception as error:
        return "ERROR", 50, str(error)
