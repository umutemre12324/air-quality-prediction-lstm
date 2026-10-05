from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from sklearn.preprocessing import MinMaxScaler

from model import AirQualityLSTM, FEATURE_COLUMNS, TARGET_COLUMN, WINDOW_SIZE

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "lstm_model.pth"
SCALER_PATH = PROJECT_DIR / "scaler.pkl"

app = FastAPI(
    title="Hava Kalitesi Tahmin Servisi",
    description=(
        f"Son {WINDOW_SIZE} saatlik ölçümden sonraki saatin "
        f"{TARGET_COLUMN} değerini tahmin eder."
    ),
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    observations: list[list[float]]

    @field_validator("observations")
    @classmethod
    def validate_observations(cls, observations: list[list[float]]) -> list[list[float]]:
        if len(observations) != WINDOW_SIZE:
            raise ValueError(f"observations tam olarak {WINDOW_SIZE} satır içermeli.")
        for index, row in enumerate(observations):
            if len(row) != len(FEATURE_COLUMNS):
                raise ValueError(
                    f"observations[{index}] tam olarak "
                    f"{len(FEATURE_COLUMNS)} özellik içermeli."
                )
        if not np.isfinite(observations).all():
            raise ValueError("Tüm gözlem değerleri sonlu sayılar olmalı.")
        return observations


@lru_cache(maxsize=1)
def load_predictor() -> tuple[AirQualityLSTM, MinMaxScaler]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model dosyası bulunamadı. Önce preprocessing.py ve ardından "
            "train.py çalıştırarak lstm_model.pth dosyasını oluşturun."
        )
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Ölçekleyici dosyası bulunamadı: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    if getattr(scaler, "n_features_in_", None) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"scaler.pkl {len(FEATURE_COLUMNS)} özellik bekliyor olmalı."
        )
    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != FEATURE_COLUMNS:
        raise ValueError("scaler.pkl özellik sırası model özellikleriyle uyuşmuyor.")

    model = AirQualityLSTM()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


@app.get("/health")
def health() -> dict[str, object]:
    model_ready = MODEL_PATH.is_file() and SCALER_PATH.is_file()
    return {
        "status": "ok",
        "model_ready": model_ready,
        "target": TARGET_COLUMN,
        "window_size": WINDOW_SIZE,
        "feature_count": len(FEATURE_COLUMNS),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float | str]:
    try:
        model, scaler = load_predictor()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    observations = np.asarray(request.observations, dtype=np.float32)
    feature_frame = pd.DataFrame(observations, columns=FEATURE_COLUMNS)
    normalized = scaler.transform(feature_frame).astype(np.float32)
    model_input = torch.from_numpy(normalized).unsqueeze(0)

    with torch.no_grad():
        prediction = float(model(model_input).item())

    inverse_input = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32)
    inverse_input[0, 0] = prediction
    prediction_value = float(scaler.inverse_transform(inverse_input)[0, 0])
    return {"target": TARGET_COLUMN, "prediction": prediction_value}
