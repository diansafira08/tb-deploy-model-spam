# train_and_compare.py
# =========================================================
# Train + Compare model (NB, LR, SVM) + heatmap + save model
# Output:
# - heatmap.png
# - svm_model.pkl
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

# Stopword Indonesia sederhana (opsional, biar Indo lebih nyambung)
INDO_STOPWORDS = {
    "yang","dan","di","ke","dari","ini","itu","untuk","dengan","pada","atau","saya","kamu","dia",
    "kami","kita","mereka","apa","mana","kok","juga","lagi","udah","sudah","belum","tidak","nggak",
    "iya","ya","aja","nih","deh","lah","dong","kan","nya","dalam","sebagai","karena","agar","bisa"
}

def guess_text_label_columns(df: pd.DataFrame):
    """
    Coba tebak kolom label & teks dari berbagai format dataset.
    Mendukung:
    - v1/v2 (SMS spam collection)
    - Category/Message
    - label/text
    - kelas/sms
    - dan beberapa variasi umum
    """
    cols = [c.strip() for c in df.columns]
    lower_map = {c: c.lower().strip() for c in df.columns}

    # kandidat label & text
    label_candidates = {"v1","label","category","kelas","class","target","y"}
    text_candidates  = {"v2","text","message","sms","content","pesan","kalimat","body","x"}

    label_col = None
    text_col = None

    for c in df.columns:
        if lower_map[c] in label_candidates:
            label_col = c
            break

    for c in df.columns:
        if lower_map[c] in text_candidates:
            text_col = c
            break

    # fallback: ambil kolom object paling masuk akal
    if label_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if obj_cols:
            label_col = obj_cols[0]

    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if len(obj_cols) >= 2:
            text_col = obj_cols[1]

    if label_col is None or text_col is None:
        raise ValueError(f"Gagal deteksi kolom label/text. Kolom dataset: {list(df.columns)}")

    return label_col, text_col

def load_dataset():
    # encoding aman (sering latin-1 kalau dari spam collection)
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")

    label_col, text_col = guess_text_label_columns(df)

    df = df[[label_col, text_col]].rename(columns={label_col:"Category", text_col:"Message"})
    df["Category"] = df["Category"].astype(str).str.lower().str.strip()
    df["Message"]  = df["Message"].astype(str)

    # normalisasi label (kalau dataset pakai 0/1 atau label aneh)
    df["Category"] = df["Category"].replace({
        "0": "ham", "1": "spam",
        "nonspam": "ham", "not spam": "ham",
        "bukan spam": "ham"
    })

    # pastikan cuma 2 kelas
    df = df[df["Category"].isin(["ham","spam"])].copy()
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
    X = df["Message"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ✅ Untuk dataset Indo: jangan pakai stop_words="english"
    # Kita pakai stopwords indo sederhana + TF-IDF ngram.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(INDO_STOPWORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Linear SVM": SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42),
    }

    results = {}
    pipes = {}

    print("=== Training & Evaluasi ===")
    for name, clf in models.items():
        pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        pipe.fit(X_train, y_train)

        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        pipes[name] = pipe

        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"- {k}: {v:.4f}")

    # heatmap
    metric_names = ["Accuracy","Precision","Recall","F1"]
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

    best = max(results, key=lambda m: results[m]["F1"])
    with open("svm_model.pkl", "wb") as f:
        pickle.dump(pipes[best], f)

    print(f"✅ Model terbaik (F1): {best}")
    print("✅ svm_model.pkl dibuat")

if __name__ == "__main__":
    main()
