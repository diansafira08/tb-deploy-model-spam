# train_and_compare.py
# =========================================================
# Tujuan:
# 1) Baca dataset spam.csv (punyamu)
# 2) Bandingkan beberapa model (NB, LR, SVM) -> metrik evaluasi
# 3) Buat heatmap evaluasi (buat bahan presentasi)
# 4) Simpan model terbaik ke file: svm_model.pkl (dipakai FastAPI)
# =========================================================

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_PATH = "spam.csv"  # file dataset kamu (taruh di root folder)


def load_and_clean_dataset():
    """
    Dataset spam.csv yang umum biasanya kolom:
    - v1: label (ham/spam)
    - v2: text pesan
    dan ada kolom kosong: Unnamed: 2, Unnamed: 3, Unnamed: 4
    Encoding sering latin-1, jadi kita pakai encoding="latin-1".
    """
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # ambil hanya 2 kolom utama
    df = df[["v1", "v2"]].rename(columns={"v1": "Category", "v2": "Message"})

    # pastikan tidak ada null
    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df["Message"] = df["Message"].astype(str)

    return df


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    # pos_label="spam" karena labelnya "spam" dan "ham"
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "F1": f1_score(y_test, y_pred, pos_label="spam", zero_division=0),
    }


def main():
    df = load_and_clean_dataset()

    X = df["Message"]
    y = df["Category"]

    # split data (stratify biar proporsi spam/ham tetap)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF (mirip materi: ngram 1-2, stopwords english, dll)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        # probability=True agar nanti FastAPI bisa tampilkan confidence (predict_proba)
        "Linear SVM": SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        ),
    }

    results = {}
    pipelines = {}

    print("=== Training & Evaluasi Model ===")
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)

        metrics = evaluate_model(pipe, X_test, y_test)
        results[name] = metrics
        pipelines[name] = pipe

        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"- {k}: {v:.4f}")

    # ===== buat heatmap =====
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    model_names = list(results.keys())
    matrix = np.array([[results[m][k] for k in metric_names] for m in model_names])

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)

    # tulis angka di kotak heatmap
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")

    ax.set_title("Heatmap Perbandingan Evaluasi Model")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=200)
    print("\n✅ heatmap.png berhasil dibuat (buat bahan presentasi)")

    # ===== pilih model terbaik berdasarkan F1 =====
    best_name = max(results, key=lambda m: results[m]["F1"])
    best_model = pipelines[best_name]
    print(f"✅ Model terbaik berdasarkan F1: {best_name}")

    # simpan model terbaik ke svm_model.pkl (nama file ikut materi)
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("✅ svm_model.pkl berhasil dibuat (dipakai oleh FastAPI)")


if __name__ == "__main__":
    main()
