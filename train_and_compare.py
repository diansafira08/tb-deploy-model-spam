# train_and_compare.py
# =========================================================
# Train + Compare model (NB, LR, SVM) + heatmap + save model
# Output:
# - heatmap.png
# - svm_model.pkl
#
# PENTING:
# - File dataset diletakkan di root project dengan nama: spam.csv
# - Script ini otomatis mencoba deteksi kolom label & teks
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

DATA_PATH = "spam.csv"


def guess_text_label_columns(df: pd.DataFrame):
    """
    Deteksi kolom label & teks dari berbagai format dataset.
    Mendukung variasi umum:
    - v1/v2 (spam collection klasik)
    - Category/Message
    - label/text
    - kelas/sms
    - target/pesan, dll
    """
    # mapping lowercase name
    lower = {c: str(c).lower().strip() for c in df.columns}

    label_candidates = {"v1", "label", "category", "kelas", "class", "target", "y", "kategori", "status"}
    text_candidates  = {"v2", "text", "message", "sms", "content", "pesan", "kalimat", "body", "x", "isi"}

    label_col = None
    text_col = None

    # cari label
    for c in df.columns:
        if lower[c] in label_candidates:
            label_col = c
            break

    # cari teks
    for c in df.columns:
        if lower[c] in text_candidates:
            text_col = c
            break

    # fallback: ambil 2 kolom object pertama
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if label_col is None and len(obj_cols) >= 1:
        label_col = obj_cols[0]
    if text_col is None and len(obj_cols) >= 2:
        text_col = obj_cols[1]

    if label_col is None or text_col is None:
        raise ValueError(
            f"Gagal deteksi kolom label/text.\n"
            f"Kolom yang ada: {list(df.columns)}\n"
            f"Saran: pastikan ada kolom label dan kolom teks."
        )

    return label_col, text_col


def normalize_labels(series: pd.Series) -> pd.Series:
    """
    Normalisasi label jadi hanya: 'spam' atau 'ham'
    Mendukung label:
    - spam/ham
    - 1/0
    - true/false
    - ya/tidak, dll (kalau ada)
    """
    s = series.astype(str).str.lower().str.strip()

    # mapping umum
    mapping = {
        "1": "spam",
        "0": "ham",
        "spam": "spam",
        "ham": "ham",
        "nonspam": "ham",
        "non-spam": "ham",
        "not spam": "ham",
        "bukan spam": "ham",
        "normal": "ham",
        "legit": "ham",
        "phishing": "spam",
        "penipuan": "spam",
        "promo": "spam",
    }

    s = s.replace(mapping)
    return s


def load_dataset() -> pd.DataFrame:
    # coba utf-8 dulu, kalau gagal fallback latin-1
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")

    label_col, text_col = guess_text_label_columns(df)

    df = df[[label_col, text_col]].rename(columns={label_col: "Category", text_col: "Message"})
    df["Category"] = normalize_labels(df["Category"])
    df["Message"] = df["Message"].astype(str)

    # ambil hanya spam/ham
    df = df[df["Category"].isin(["spam", "ham"])].copy()

    # drop empty text
    df["Message"] = df["Message"].str.strip()
    df = df[df["Message"].str.len() > 0].copy()

    return df


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label="spam", zero_division=0),
        "F1": f1_score(y_test, y_pred, pos_label="spam", zero_division=0),
    }


def main():
    df = load_dataset()

    # ====== PRINT CEK DATASET (INI YANG KAMU MAU) ======
    print("\n===== CEK DATASET (5 BARIS PERTAMA) =====")
    print(df.head(5))
    print("\n===== DISTRIBUSI LABEL =====")
    print(df["Category"].value_counts())

    X = df["Message"]
    y = df["Category"]

    # stratify biar proporsi spam/ham tetap
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ PENTING UNTUK INDONESIA:
    # JANGAN pakai stop_words="english"
    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Linear SVM": SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        ),
    }

    results = {}
    pipes = {}

    print("\n=== Training & Evaluasi ===")
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)

        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        pipes[name] = pipe

        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"- {k}: {v:.4f}")

    # ===== heatmap =====
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    model_names = list(results.keys())
    matrix = np.array([[results[m][k] for k in metric_names] for m in model_names])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")

    ax.set_title("Heatmap Perbandingan Evaluasi Model")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=200)
    print("\n✅ heatmap.png dibuat")

    # pilih terbaik berdasarkan F1
    best = max(results, key=lambda m: results[m]["F1"])
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(pipes[best], f)

    print(f"✅ Model terbaik (F1): {best}")
    print("✅ svm_model.pkl dibuat (model terbaru)")


if __name__ == "__main__":
    main()
