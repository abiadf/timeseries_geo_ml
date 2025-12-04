import torch
import numpy as np
from typing import List, Callable, Tuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        """Ensure the number of splits does not exceed the size of the dimension being split."""
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

    @staticmethod
    def to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Ensure numpy → torch.float32."""
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=torch.float32)


class HorizontalFedEncoder(BaseFederatedEncoder):
    """Split along samples/pages. y must be split the same way."""
    def __init__(self, num_splits: int):
        super().__init__(num_splits, dim_splitting=0)

    def encode_federated(self, X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor,
            encoder_builder: Callable, fit_function: Callable, batch_size: int):# -> Tuple[np.ndarray, np.ndarray]:
        """Encode data in a horizontal manner. Outputs latents"""
        X_train = self.to_tensor(X_train)
        X_test  = self.to_tensor(X_test)
        y_train = self.to_tensor(y_train)
        y_test  = self.to_tensor(y_test)

        self._adjust_splits(X_train)
        Xtr_splits = self._split(X_train)
        Xte_splits = self._split(X_test)
        encoders, Z_train_list, Z_test_list = [], [], []

        for i in range(self.num_splits):
            enc = encoder_builder()
            fit_function(enc, Xtr_splits[i])
            encoders.append(enc)
            Z_train_list.append(self._encode_in_batches(enc, Xtr_splits[i], bs=batch_size))
            Z_test_list.append(self._encode_in_batches(enc, Xte_splits[i],  bs=batch_size))

        return Z_train_list, Z_test_list
        # Z_train_cat = torch.cat(Z_train, dim=0).numpy()
        # Z_test_cat  = torch.cat(Z_test, dim=0).numpy()
        # return Z_train_cat, Z_test_cat

    def combine_latents(self, Z_train_list, Z_test_list) -> Tuple[np.ndarray, np.ndarray]:
        """Combine latents from different splits by concat along sample dimension"""
        Z_train_cat = torch.cat(Z_train_list, dim=0).numpy()
        Z_test_cat  = torch.cat(Z_test_list, dim=0).numpy()
        return Z_train_cat, Z_test_cat


class VerticalFedEncoder(BaseFederatedEncoder):
    """Split along feature dimension. y is shared across all splits."""
    def __init__(self, num_splits: int):
        super().__init__(num_splits, dim_splitting=2)

    def encode_federated(self, X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor,
            encoder_builder: Callable, fit_fn: Callable, batch_size: int): # -> Tuple[np.ndarray, np.ndarray]:
        """Encode data in a vertical manner. Outputs latents"""
        X_train = self.to_tensor(X_train)
        X_test  = self.to_tensor(X_test)

        self._adjust_splits(X_train)
        Xtr_splits = self._split(X_train)
        Xte_splits = self._split(X_test)
        encoders, Z_train_list, Z_test_list = [], [], []

        for i in range(self.num_splits):
            enc = encoder_builder()
            fit_fn(enc, Xtr_splits[i])
            encoders.append(enc)
            Z_train_list.append(self._encode_in_batches(enc, Xtr_splits[i], bs=batch_size))
            Z_test_list.append(self._encode_in_batches(enc, Xte_splits[i],  bs=batch_size))

        return Z_train_list, Z_test_list
        # Z_train_cat = torch.cat(Z_train_list, dim=1).numpy()
        # Z_test_cat  = torch.cat(Z_test_list, dim=1).numpy()
        # return Z_train_cat, Z_test_cat

    def combine_latents(self, Z_train_list, Z_test_list) -> Tuple[np.ndarray, np.ndarray]:
        """Combine latents from different splits by concat along sample dimension"""
        Z_train_cat = torch.cat(Z_train_list, dim=0).numpy()
        Z_test_cat  = torch.cat(Z_test_list, dim=0).numpy()
        return Z_train_cat, Z_test_cat

