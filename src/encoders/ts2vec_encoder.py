""""TS2Vec encoder backend (calls the TS2Vec class) + early stopping"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset 

from src.ts2vec_model.ts2vec import TS2Vec # the real ts2vec library
from src.utils.model_utils import profile_epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TS2VecEncoder:
    """TS2Vec + Linear Predictor pipeline with externally set hyperparameters
       Supports early stopping during TS2Vec training."""

    def __init__(self, lr, z_pooling: str = "mean", device=None, patience: int = 20, max_train_length: int = 512):
        """Args:
            z_pooling (str): Pooling method for encoding.
            device (str or torch.device): "cpu" or "cuda".
            patience (int): Number of epochs with no improvement before stopping"""
        self.z_pooling   = z_pooling
        self.ts_model    = None
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.patience    = patience
        self._stop_early = False  # flag to indicate if early stopping occurred
        self.lr          = lr
        self.max_train_length = max_train_length

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

    def fit_ts2vec(self, X_train, hidden_dims=12, output_dims=8, depth=5, batch_size=16, n_epochs=20):
        """Fit TS2Vec on training data with early stopping."""
        stopper       = self.EarlyStopper(self.patience)
        self.ts_model = TS2Vec(input_dims=X_train.shape[2], hidden_dims=hidden_dims, depth=depth, lr=self.lr,
                               output_dims=output_dims, batch_size=batch_size, device=self.device,
                               after_epoch_callback=stopper, max_train_length=self.max_train_length, temporal_unit=1)
        self.ts_model.fit(X_train.astype(np.float32), n_epochs=n_epochs, verbose=True)
        
        # ========= profiling ==========
        class DummyLoss(torch.nn.Module):
            def forward(self, out, target=None):
                return out.sum()

        criterion         = DummyLoss()
        train_dataset     = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
        train_loader      = DataLoader(train_dataset, batch_size=self.ts_model.batch_size, shuffle=False)
        optimizer         = torch.optim.AdamW(self.ts_model._net.parameters(), lr=1e-3)
        profiling_metrics = profile_epoch(self.ts_model._net, train_loader, optimizer, criterion, device=self.device)
        return profiling_metrics
        # ==============================

        # # TS2Vec expects numpy arrays in .fit(), so feed batches sequentially
        # for epoch in range(n_epochs):
        #     for batch in loader:
        #         batch_np = batch[0].cpu().numpy() if self.device != "cpu" else batch[0].numpy()
        #         self.ts_model.fit(batch_np, n_epochs=1, verbose=True)
        #     if stopper.stop:
        #         print(f"Stopped early at epoch {epoch+1}")
        #         break
        # self._stop_early = stopper.stop

        # X_train = X_train.astype(np.float32)
        # stopper = self.EarlyStopper(self.patience)
        # self.ts_model = TS2Vec(
        #     input_dims=X_train.shape[2],
        #     hidden_dims=hidden_dims,
        #     depth=depth,
        #     output_dims=output_dims,
        #     batch_size=batch_size,
        #     device=self.device,
        #     after_epoch_callback=stopper
        # )
        # self.ts_model.fit(X_train, n_epochs=n_epochs, verbose=True)
        # self._stop_early = stopper.stop  # record whether early stopping triggered

    def encode(self, X, pooling=None):
        """Encode X into latent embeddings with pooling (TS2Vec expects NumPy arrays)."""
        if self.ts_model is None:
            raise ValueError("TS2Vec model not trained. Call fit first.")

        # Ensure input is a NumPy array
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X = X.astype(np.float32)
        z = self.ts_model.encode(X)  # shape: (B, T, latent_dim)

        # pooling using NumPy (TS2Vec returns NumPy array)
        p = pooling or self.z_pooling
        if pooling is None:
            return z  # **return full time embeddings**
        elif p == "mean":
            return z.mean(axis=1)
        elif p == "max":
            return z.max(axis=1)
        elif p == "last":
            return z[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {p}")

