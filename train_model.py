import pickle
import re
import string
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


STOP_WORDS = set(sklearn_text.ENGLISH_STOP_WORDS)
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "News_Dataset"
FAKE_DATASET_PATH = DATASET_DIR / "Fake.csv"
TRUE_DATASET_PATH = DATASET_DIR / "True.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(words)


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    if not FAKE_DATASET_PATH.exists() or not TRUE_DATASET_PATH.exists():
        raise FileNotFoundError(
            "News_Dataset/Fake.csv and News_Dataset/True.csv were not found."
        )

    fake_dataset = pd.read_csv(FAKE_DATASET_PATH)
    true_dataset = pd.read_csv(TRUE_DATASET_PATH)

    fake_dataset["label"] = "FAKE"
    true_dataset["label"] = "REAL"

    dataset = pd.concat([fake_dataset, true_dataset], ignore_index=True)

    if "title" in dataset.columns and "text" in dataset.columns:
        dataset["text"] = (
            dataset["title"].fillna("").astype(str) + " " +
            dataset["text"].fillna("").astype(str)
        ).str.strip()
    elif "text" not in dataset.columns:
        raise ValueError("Dataset must contain a 'text' column, or both 'title' and 'text'.")

    dataset = dataset[["text", "label"]].dropna()
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset = dataset[dataset["text"] != ""]

    dataset["text"] = dataset["text"].apply(preprocess_text)

    texts = dataset["text"].tolist()
    labels = dataset["label"].tolist()

    print("Dataset loaded successfully")
    print("Dataset size:", len(dataset))
    print("Labels:", dataset["label"].value_counts().to_dict())

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vectors, y_train)

    # Evaluation
    predictions = model.predict(x_test_vectors)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", round(accuracy, 4))

    # Save model
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"Saved {MODEL_PATH.name}")
    print(f"Saved {VECTORIZER_PATH.name}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    train_model()
