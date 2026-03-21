import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from video_detector import extract_video_features


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "video dataset"
VIDEO_MODEL_PATH = BASE_DIR / "video_model.pkl"


def load_videos_from_folder(folder_path, label):
    rows = []
    if not folder_path.exists():
        return rows

    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            continue

        try:
            extracted = extract_video_features(str(file_path))
            if extracted[0] is None:
                continue
            rows.append((extracted[0], label))
        except Exception:
            continue

    return rows


def train_video_model():
    attack_rows = load_videos_from_folder(DATASET_DIR / "attack", "FAKE")
    real_rows = load_videos_from_folder(DATASET_DIR / "real_video", "REAL")
    dataset_rows = attack_rows + real_rows

    if not dataset_rows:
        raise ValueError("No usable videos were found inside 'video dataset'.")

    unique_labels = sorted({label for _, label in dataset_rows})
    if len(unique_labels) < 2:
        raise ValueError("Video dataset must contain both REAL and FAKE videos.")

    x = np.array([row[0] for row in dataset_rows], dtype=float)
    y = np.array([row[1] for row in dataset_rows])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Video dataset folders:")
    print(f"  attack: {len(attack_rows)}")
    print(f"  real_video: {len(real_rows)}")
    print("Video dataset size:", len(dataset_rows))
    print("Available labels:", unique_labels)
    print("Video model accuracy:", round(accuracy, 4))
    print(classification_report(y_test, predictions))

    with open(VIDEO_MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Saved {VIDEO_MODEL_PATH.name}")


if __name__ == "__main__":
    train_video_model()
