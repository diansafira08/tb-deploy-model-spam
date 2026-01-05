# main.py
# =========================================================
# FastAPI Web App untuk Spam Classifier
# - GET  /         : tampilkan halaman web (HTML)
# - POST /predict  : terima teks & kembalikan prediksi + confidence
# Bonus:
# - Ambil contoh HAM dan SPAM langsung dari dataset spam.csv,
#   jadi tombol "Contoh HAM/SPAM" tidak ngawur lagi.
# =========================================================

import os
import pickle
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Spam SMS Classifier", version="1.0.0")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


class EmailRequest(BaseModel):
    email: str


def load_examples_from_dataset():
    """
    Ambil 1 contoh ham dan 1 contoh spam dari dataset.
    Deteksi kolom secara fleksibel seperti di training.
    Kalau gagal, fallback ke teks default.
    """
    default_ham = "Halo, nanti jam 7 jadi ketemu? Jangan lupa bawa berkas ya."
    default_spam = "SELAMAT! Anda menang undian. Kirim kode OTP ke nomor ini untuk verifikasi."

    try:
        try:
            df = pd.read_csv(DATA_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(DATA_PATH, encoding="latin-1")

        # deteksi kolom label & text
        lower = {c: str(c).lower().strip() for c in df.columns}
        label_candidates = {"v1","label","category","kelas","class","target","y","kategori","status"}
        text_candidates  = {"v2","text","message","sms","content","pesan","kalimat","body","x","isi"}

        label_col = None
        text_col = None

        for c in df.columns:
            if lower[c] in label_candidates:
                label_col = c
                break
        for c in df.columns:
            if lower[c] in text_candidates:
                text_col = c
                break

        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if label_col is None and len(obj_cols) >= 1:
            label_col = obj_cols[0]
        if text_col is None and len(obj_cols) >= 2:
            text_col = obj_cols[1]

        df = df[[label_col, text_col]].rename(columns={label_col:"Category", text_col:"Message"})
        df["Category"] = df["Category"].astype(str).str.lower().str.strip()
        df["Message"] = df["Message"].astype(str).str.strip()

        # normalisasi label
        df["Category"] = df["Category"].replace({
            "1":"spam","0":"ham","nonspam":"ham","non-spam":"ham","not spam":"ham","bukan spam":"ham"
        })

        df = df[df["Category"].isin(["spam","ham"])].copy()
        df = df[df["Message"].str.len() > 0].copy()

        sample_ham = df[df["Category"] == "ham"]["Message"].iloc[0] if (df["Category"] == "ham").any() else default_ham
        sample_spam = df[df["Category"] == "spam"]["Message"].iloc[0] if (df["Category"] == "spam").any() else default_spam

        return sample_ham, sample_spam

    except Exception:
        return default_ham, default_spam


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    sample_ham, sample_spam = load_examples_from_dataset()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "sample_ham": sample_ham, "sample_spam": sample_spam}
    )


@app.post("/predict")
async def predict_email(req: EmailRequest):
    text = (req.email or "").strip()
    if not text:
        return {"prediction": "empty_input", "confidence": 0.0}

    X = [text]
    pred = model.predict(X)[0]
    conf = float(model.predict_proba(X).max())

    return {"prediction": str(pred), "confidence": round(conf, 4)}
