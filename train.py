import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import AirQualityLSTM


def main() -> None:
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    model = AirQualityLSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = []

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss_value = loss_function(output, y_train)
        loss_value.backward()
        optimizer.step()

        loss.append(loss_value.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/100], Loss: {loss_value.item():.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.close()

    torch.save(model.state_dict(), "lstm_model.pth")
    print("Model weights saved to lstm_model.pth")


if __name__ == "__main__":
    main()