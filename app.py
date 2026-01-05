# app.py
# =========================================================
# FastAPI Web App untuk Spam Email Classifier
# - GET  /         : tampilkan halaman web (HTML)
# - POST /predict  : terima teks email dan kembalikan prediksi + confidence
#
# File model: svm_model.pkl (sudah kamu buat)
# =========================================================

import os
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- Path aman untuk Windows / Railway ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")

# --- Load model ---
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(
    title="Spam Email Classification",
    version="1.0.0"
)

# --- Template folder ---
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- Schema request/response ---
class EmailRequest(BaseModel):
    email: str

class EmailResponse(BaseModel):
    prediction: str
    confidence: float

# --- Route home (UI) ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Route predict ---
@app.post("/predict", response_model=EmailResponse)
async def predict_email(req: EmailRequest):
    text = req.email.strip()

    # kalau kosong, balikin default
    if not text:
        return {"prediction": "empty_input", "confidence": 0.0}

    X = [text]
    pred = model.predict(X)[0]

    # Linear SVM kamu dibuat probability=True, jadi predict_proba bisa dipakai
    conf = float(model.predict_proba(X).max())

    return {
        "prediction": str(pred),
        "confidence": round(conf, 4)
    }
