import torch
import numpy as np
from typing import List, Callable, Tuple

class BaseFederatedEncoder:
    """Shared utilities for horizontal/vertical FL.
    Splits either along samples (horizontal) or features (vertical)
    0 = splits along pages (horiz)
    1 = splits along rows (not a thing)
    2 = splits along features (vertical)"""
    def __init__(self, num_splits: int, dim_splitting: int):
        self.num_splits    = num_splits
        self.dim_splitting = dim_splitting

    def _adjust_splits(self, X: torch.Tensor):
        max_splits = X.shape[self.dim_splitting]
        if self.num_splits > max_splits:
            print(f"⚠️ num_splits ({self.num_splits}) > dim_size ({max_splits}) → adjusting.")
            self.num_splits = max_splits

    def _split(self, X: torch.Tensor) -> List[torch.Tensor]:
        return torch.tensor_split(X, self.num_splits, dim=self.dim_splitting)

    @staticmethod
    def _encode_in_batches(encoder, X: torch.Tensor, bs: int = 64) -> torch.Tensor:
        """Encode X via minibatches."""
        outs = []
        arr  = X.cpu().numpy()
        for i in range(0, len(arr), bs):
            z = encoder.encode(arr[i:i+bs])
            outs.append(z)
        z    = np.concatenate(outs, axis=0)
        return torch.tensor(z.reshape(z.shape[0], -1), dtype=torch.float32)

class HorizontalFedEncoder(BaseFederatedEncoder):
    """Split along samples/pages. y must be split the same way."""
    def __init__(self, num_splits: int):
        super().__init__(num_splits, dim_splitting=0)

    def run(
        self,
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        encoder_builder: Callable,  # returns new encoder
        fit_fn: Callable,            # fit_fn(encoder, X_split)
        encode_fn: Callable          # encode_fn(encoder, X_split)
    ) -> Tuple[np.ndarray, np.ndarray]:

        self._adjust_splits(X_train)

        Xtr_splits = self._split(X_train)
        Xte_splits = self._split(X_test)
        ytr_splits = self._split(y_train)
        yte_splits = self._split(y_test)

        encoders, Ztr, Zte = [], [], []

        for i in range(self.num_splits):
            enc = encoder_builder()
            fit_fn(enc, Xtr_splits[i])
            encoders.append(enc)

            Ztr.append(encode_fn(enc, Xtr_splits[i]))
            Zte.append(encode_fn(enc, Xte_splits[i]))

        Z_train = torch.cat(Ztr, dim=0).numpy()
        Z_test  = torch.cat(Zte, dim=0).numpy()
        return Z_train, Z_test


class VerticalFedEncoder(BaseFederatedEncoder):
    """Split along feature dimension. y is shared across all splits."""
    def __init__(self, num_splits: int):
        super().__init__(num_splits, dim_splitting=2)

    def run(
        self,
        X_train: torch.Tensor,
        X_test: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        encoder_builder: Callable,
        fit_fn: Callable,
        encode_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:

        self._adjust_splits(X_train)
        Xtr_splits = self._split(X_train)
        Xte_splits = self._split(X_test)

        encoders, Z_train, Z_test = [], [], []

        for i in range(self.num_splits):
            enc = encoder_builder()
            fit_fn(enc, Xtr_splits[i])
            encoders.append(enc)

            Z_train.append(encode_fn(enc, Xtr_splits[i]))
            Z_test.append(encode_fn(enc, Xte_splits[i]))

        Z_train = torch.cat(Z_train, dim=1).numpy()
        Z_test  = torch.cat(Z_test, dim=1).numpy()
        return Z_train, Z_test
