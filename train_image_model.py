import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from image_detector import extract_image_features


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_MODEL_PATH = BASE_DIR / "image_model.pkl"


def load_images_from_folder(folder_path, label):
    rows = []
    if not folder_path.exists():
        return rows

    for file_path in folder_path.iterdir():
        if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        try:
            image = Image.open(file_path)
            features = extract_image_features(image)
            rows.append((features, label))
        except Exception:
            continue

    return rows


def collect_dataset_rows():
    dataset_rows = []
    folder_counts = {}

    for split_name in ["Train", "Test"]:
        for class_name, label in [("Real", "REAL"), ("Fake", "FAKE")]:
            folder_path = DATASET_DIR / split_name / class_name
            rows = load_images_from_folder(folder_path, label)
            dataset_rows.extend(rows)
            folder_counts[f"{split_name}/{class_name}"] = len(rows)

    return dataset_rows, folder_counts


def balance_dataset_rows(dataset_rows):
    fake_rows = [row for row in dataset_rows if row[1] == "FAKE"]
    real_rows = [row for row in dataset_rows if row[1] == "REAL"]

    if not fake_rows or not real_rows:
        return dataset_rows

    target_count = min(len(fake_rows), len(real_rows))
    rng = np.random.default_rng(42)

    fake_indices = rng.choice(len(fake_rows), size=target_count, replace=False)
    real_indices = rng.choice(len(real_rows), size=target_count, replace=False)

    balanced_rows = [fake_rows[index] for index in fake_indices]
    balanced_rows.extend(real_rows[index] for index in real_indices)
    rng.shuffle(balanced_rows)
    return balanced_rows


def train_image_model():
    dataset_rows, folder_counts = collect_dataset_rows()

    if not dataset_rows:
        raise ValueError("No usable images were found inside dataset/Train or dataset/Test.")

    balanced_rows = balance_dataset_rows(dataset_rows)

    x = np.array([row[0] for row in balanced_rows], dtype=float)
    y = np.array([row[1] for row in balanced_rows])

    unique_labels = sorted(set(y.tolist()))
    if len(unique_labels) < 2:
        raise ValueError("Image dataset must contain both REAL and FAKE images.")

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

    print("Image dataset folders:")
    for folder_name, count in folder_counts.items():
        print(f"  {folder_name}: {count}")
    print("Image dataset size:", len(dataset_rows))
    print("Balanced training size:", len(balanced_rows))
    print("Available labels:", unique_labels)
    print("Image model accuracy:", round(accuracy, 4))
    print(classification_report(y_test, predictions))

    with open(IMAGE_MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Saved {IMAGE_MODEL_PATH.name}")


if __name__ == "__main__":
    train_image_model()
