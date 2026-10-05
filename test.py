from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from model import AirQualityLSTM, FEATURE_COLUMNS, TARGET_COLUMN, WINDOW_SIZE

PROJECT_DIR = Path(__file__).resolve().parent


def inverse_transform_target(
    values: np.ndarray, scaler: MinMaxScaler
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    padded = np.zeros((len(values), len(FEATURE_COLUMNS)), dtype=np.float32)
    padded[:, 0] = values[:, 0]
    frame = pd.DataFrame(padded, columns=FEATURE_COLUMNS)
    return scaler.inverse_transform(frame)[:, :1]


def main() -> None:
    x_test = np.load(PROJECT_DIR / "x_test.npy")
    y_test = np.load(PROJECT_DIR / "y_test.npy")
    model_path = PROJECT_DIR / "lstm_model.pth"
    scaler_path = PROJECT_DIR / "scaler.pkl"

    if not model_path.is_file():
        raise FileNotFoundError(
            "Model bulunamadı. Önce preprocessing.py ve train.py çalıştırın."
        )
    if not scaler_path.is_file():
        raise FileNotFoundError(
            "scaler.pkl bulunamadı. Önce preprocessing.py çalıştırın."
        )
    if x_test.ndim != 3 or x_test.shape[1:] != (
        WINDOW_SIZE,
        len(FEATURE_COLUMNS),
    ):
        raise ValueError(
            f"x_test şekli (örnek, {WINDOW_SIZE}, {len(FEATURE_COLUMNS)}) olmalı; "
            f"bulunan şekil: {x_test.shape}"
        )

    scaler = joblib.load(scaler_path)
    if getattr(scaler, "n_features_in_", None) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"scaler.pkl {len(FEATURE_COLUMNS)} özellik içermelidir. "
            "preprocessing.py dosyasını yeniden çalıştırın."
        )

    model = AirQualityLSTM()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        predictions = model(torch.tensor(x_test, dtype=torch.float32)).numpy()

    predictions_inverse = inverse_transform_target(predictions, scaler)
    actual_inverse = inverse_transform_target(y_test, scaler)
    if len(predictions_inverse) != len(actual_inverse):
        raise ValueError("Tahmin ve gerçek değer sayıları birbiriyle uyuşmuyor.")

    rmse = np.sqrt(mean_squared_error(actual_inverse, predictions_inverse))
    print(f"{TARGET_COLUMN} RMSE: {rmse:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(actual_inverse, label="Gerçek")
    plt.plot(predictions_inverse, label="Tahmin")
    plt.xlabel("Örnek")
    plt.ylabel(TARGET_COLUMN)
    plt.title(f"Gerçek ve Tahmin Edilen {TARGET_COLUMN}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "actual_vs_predicted.png")
    plt.close()
    print("Grafik actual_vs_predicted.png dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
