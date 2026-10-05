import torch
import torch.nn as nn

FEATURE_COLUMNS = [
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "hour",
    "month",
]
TARGET_COLUMN = "CO(GT)"
WINDOW_SIZE = 72
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class AirQualityLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = x.new_zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE)
        c0 = x.new_zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE)
        output, _ = self.lstm(x, (h0, c0))
        return self.fc(output[:, -1, :])
