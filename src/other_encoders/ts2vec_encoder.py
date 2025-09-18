
import numpy as np
import torch
import torch.nn as nn

from ts2vec import TS2Vec # library

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TS2VecEncoder:
    """TS2Vec + Linear Predictor pipeline with externally set hyperparameters
       Supports early stopping during TS2Vec training."""

    def __init__(self, z_pooling: str = "mean", device="cpu", patience: int = 20):
        """Args:
            z_pooling (str): Pooling method for encoding.
            device (str or torch.device): "cpu" or "cuda".
            patience (int): Number of epochs with no improvement before stopping"""
        self.z_pooling = z_pooling
        self.ts_model  = None
        self.device    = device
        self.patience  = patience
        self._stop_early = False  # flag to indicate if early stopping occurred

    class EarlyStopper:
        """Callback for TS2Vec early stopping."""
        def __init__(self, patience: int):
            self.patience = patience
            self.best = float("inf")
            self.wait = 0
            self.stop = False

        def __call__(self, model, loss: float):
            if loss < self.best:
                self.best = loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Early stopping at epoch {model.n_epochs}")
                    self.stop = True
                    model.n_epochs = 1e9  # forces loop exit

    def fit(self, X_train, hidden_dims=12, output_dims=8, depth=5, batch_size=16, n_epochs=20):
        """Fit TS2Vec on training data with early stopping."""
        X_train = X_train.astype(np.float32)
        stopper = self.EarlyStopper(self.patience)
        self.ts_model = TS2Vec(
            input_dims=X_train.shape[2],
            hidden_dims=hidden_dims,
            depth=depth,
            output_dims=output_dims,
            batch_size=batch_size,
            device=self.device,
            after_epoch_callback=stopper
        )
        self.ts_model.fit(X_train, n_epochs=n_epochs, verbose=True)
        self._stop_early = stopper.stop  # record whether early stopping triggered

    def encode(self, X, pooling=None):
        """Encode X into latent embeddings with pooling."""
        if self.ts_model is None:
            raise ValueError("TS2Vec model not trained. Call fit first.")

        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        z = self.ts_model.encode(X.astype(np.float32))

        p = pooling or self.z_pooling
        if p == "mean":
            return z.mean(axis=1)
        elif p == "max":
            return z.max(axis=1)
        elif p == "last":
            return z[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {p}")


# class TS2VecTorchWrapper(nn.Module):
#     """Wrap TS2Vec model for use as a torch.nn.Module so we can access its params"""
#     def __init__(self, ts_model):
#         super().__init__()
#         self.net = ts_model.net   # grab the internal network (nn.Module)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)

