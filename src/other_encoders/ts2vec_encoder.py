
import numpy as np
import torch
import torch.nn as nn

from ts2vec import TS2Vec # library

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TS2VecEncoder:
    """TS2Vec + Linear Predictor pipeline with externally set hyperparameters
        1. fit_ts2vec(X_train, X_test)
        2. convert_to_tensors(z_train, z_test, y_train, y_test)
        3. make_predictor(latent_dim, y_dim)
        4. train_predictor(...)
        5. evaluate(...)"""

    def __init__(self, z_pooling: str = "mean", device="cpu"):
        self.z_pooling = z_pooling
        self.ts_model  = None
        self.device    = device

    def fit(self, X_train, hidden_dims=12, output_dims=8, depth=5, batch_size=16, n_epochs=20):
        """Fit TS2Vec on training data (unsupervised)."""
        X_train       = X_train.astype(np.float32)
        self.ts_model = TS2Vec(input_dims=X_train.shape[2],
                               hidden_dims=hidden_dims,
                               depth=depth,
                               output_dims=output_dims,
                               batch_size=batch_size,
                               device=self.device)
        self.ts_model.fit(X_train, n_epochs=n_epochs, verbose=True)

    def encode(self, X, pooling=None):
        """Encode X into latent embeddings with pooling (accepts NumPy array or torch tensor)."""
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

