import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    """LSTM for sequence regression."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last timestep

class LSTMTrainer:
    """Simple training + validation wrapper for PyTorch models."""
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, device: str = "cpu"):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

    def fit(self, train_X, train_y, val_X, val_y, epochs: int = 20):
        train_X, train_y = train_X.to(self.device), train_y.to(self.device)
        val_X, val_y     = val_X.to(self.device), val_y.to(self.device)

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(train_X)
            loss   = self.criterion(output, train_y)
            loss.backward()
            self.optimizer.step()

            # --- Validate ---
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_X)
                val_loss   = self.criterion(val_output, val_y)

            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | train loss={loss.item():.4f} | val loss={val_loss.item():.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))

