import torch
import torch.nn as nn
import torch.optim as optim
from methods.forecasting_module import BaseForecaster


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

# moved from previously unused part of the notebok code
class Seq2SeqLSTM(nn.Module):
    """
    Encoder-decoder LSTM for multi-step forecasting.
    Auto-regressive: generates one step at a time.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Decoder LSTM (one step at a time)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, horizon):
        """
        encoder_input: (batch, seq_len, input_size)
        horizon: number of future steps to predict
        returns: (batch, horizon, output_size)
        """
        batch_size = encoder_input.size(0)
        # Encode
        _, (h, c) = self.encoder(encoder_input)

        # Initialize decoder input as last step of encoder
        decoder_input = encoder_input[:, -1:, :]  # shape: (batch, 1, input_size)
        outputs = []

        for t in range(horizon):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            step_pred = self.fc(out)  # (batch, 1, output_size)
            outputs.append(step_pred)
            decoder_input = step_pred  # feed prediction as next input

        return torch.cat(outputs, dim=1)  # (batch, horizon, output_size)

# moved from previously unused part of the notebok code
class FlexibleLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last step only

# moved from previously unused part of the notebok code
class FlexibleLSTMTrainer:
    def __init__(self, model, lr=1e-3, epochs=30, device="cpu"):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.loss_fn = nn.MSELoss()

    def _make_batches(self, y, X, seq_len, horizon):
        """Convert timeseries to (X_seq, y_future) windows."""
        data = y if X is None else np.concatenate([y, X], axis=1)
        X_batches, y_batches = [], []
        for i in range(len(data) - seq_len - horizon):
            X_batches.append(data[i:i+seq_len])
            y_batches.append(y[i+seq_len:i+seq_len+horizon])
        return (torch.tensor(np.stack(X_batches), dtype=torch.float32),
                torch.tensor(np.stack(y_batches), dtype=torch.float32),)

    def fit(self, y_train, X_train, seq_len, horizon, y_val=None, X_val=None):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        Xb, yb = self._make_batches(y_train, X_train, seq_len, horizon)
        Xb, yb = Xb.to(self.device), yb.to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            preds = self.model(Xb)
            loss  = self.loss_fn(preds, yb[:, -1, :])  # predict horizon last-step
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}, loss={loss.item():.4f}")

    def predict_seq2seq(self, y_test, X_test, seq_len, horizon):
        Xb, yb = self._make_batches(y_test, X_test, seq_len, horizon)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(Xb.to(self.device)).cpu().numpy()
        return preds

    def predict_autoreg(self, y_test, X_test, seq_len, horizon):
        """Step-by-step forecasting with feedback of predictions."""
        data  = y_test if X_test is None else np.concatenate([y_test, X_test], axis=1)
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i in range(len(data) - seq_len - horizon):
                window = torch.tensor(data[i:i+seq_len], dtype=torch.float32).unsqueeze(0).to(self.device)
                pred   = self.model(window).cpu().numpy()
                preds.append(pred)
                # feed prediction back only into y part (not exogenous)
                data[i+seq_len] = np.concatenate([pred[0], data[i+seq_len, y_test.shape[1]:]])
        return np.array(preds)


# moved from previously unused part of the notebok code
class LSTMForecaster(BaseForecaster):
    def __init__(self, df, time_col: str, y_cols: list[str],
                 use_exogenous: bool = False, seq_len: int = 30, horizon: int = 10,
                 hidden_size: int = 64, epochs: int = 30, lr: float = 1e-3):
        super().__init__(df, time_col, y_cols)
        self.use_exogenous = use_exogenous
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr

        # Exogenous columns = all columns minus time + y
        if self.use_exogenous:
            self.X_cols = [c for c in df.columns if c not in [time_col] + y_cols]
        else:
            self.X_cols = []

    def _prepare_data(self, df):
        """Return tensors for y (and X if exogenous)."""
        y = df[self.y_cols].values.astype(np.float32)
        if self.use_exogenous and self.X_cols:
            X = df[self.X_cols].values.astype(np.float32)
        else:
            X = None
        return y, X

    def fit(self, df_train, df_val=None):
        y_train, X_train = self._prepare_data(df_train)
        y_val, X_val = (None, None) if df_val is None else self._prepare_data(df_val)

        self.model = FlexibleLSTM(
            input_dim=len(self.y_cols) + (X_train.shape[1] if X_train is not None else 0),
            hidden_size=self.hidden_size,
            output_dim=len(self.y_cols))

        trainer = FlexibleLSTMTrainer(self.model, lr=self.lr, epochs=self.epochs)
        trainer.fit(y_train, X_train, self.seq_len, self.horizon,
                    y_val=y_val, X_val=X_val)
        self.trainer = trainer

    def predict(self, df_test, autoregressive: bool = False):
        y_test, X_test = self._prepare_data(df_test)
        if autoregressive:
            return self.trainer.predict_autoreg(y_test, X_test, self.seq_len, self.horizon)
        else:
            return self.trainer.predict_seq2seq(y_test, X_test, self.seq_len, self.horizon)

    def forecast_lstm(self, train_df, test_df, horizon: int,
                      use_exogenous: bool = True, stepwise: bool = False):
        """Train + forecast like SARIMAX.
        Args:
            train_df, test_df: pandas DataFrames
            horizon: forecast horizon
            use_exogenous: toggle exogenous features
            stepwise: if True → autoregressive, else → direct multi-step
        Returns:
            forecast_dict: {y_col: np.array predictions}
            y_true: true target values (scaled)"""
        # override exogenous toggle for this run
        self.use_exogenous = use_exogenous
        if use_exogenous:
            self.X_cols = [c for c in train_df.columns if c not in [self.time_col] + self.y_cols]
        else:
            self.X_cols = []

        self.fit(train_df)
        preds = self.predict(test_df, autoregressive=stepwise)
        y_true, _ = self._prepare_data(test_df)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        forecast_dict = {col: preds[:, i] for i, col in enumerate(self.y_cols)}
        self.forecast_dfs = {col: forecast_dict[col] for col in self.y_cols}
        return forecast_dict, y_true

