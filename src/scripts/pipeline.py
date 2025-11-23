
import os
import numpy as np

def split_labeled_unlabeled_data(desired_dataset, interim_data_loc, data_splitting, label_frac, X_train, y_train_scaled, X_test, y_test_scaled, params):
    """Given a data splitting method and its %, split the data into labeled and unlabeled portions."""
    n_train     = len(X_train)
    n_labeled   = int(np.ceil(label_frac * n_train))
    shuffled_idx= np.random.permutation(n_train)

    if data_splitting == "missing_labels": # Keep all of X_train, split y
        X_L = X_train[shuffled_idx[:n_labeled]]
        y_L = y_train_scaled[shuffled_idx[:n_labeled]]
        X_U = X_train[shuffled_idx[n_labeled:]]
        y_U = y_train_scaled[shuffled_idx[n_labeled:]]
    elif data_splitting == "reduced_data": # Shrink X_train & y_train by fraction, no unlabeled
        X_L = X_train[shuffled_idx[:n_labeled]]
        y_L = y_train_scaled[shuffled_idx[:n_labeled]]
        X_U = np.empty((0, *X_L.shape[1:]), dtype=X_L.dtype)
        y_U = np.empty((0, *y_L.shape[1:]), dtype=y_L.dtype)

    # For supervised training, X_train is the labeled portion ONLY
    X_train        = X_L
    y_train_scaled = y_L

    # Optional: combine for TimeVAE or other use
    X_small = np.concatenate([X_train, X_test], axis=0)
    y_small = np.concatenate([y_train_scaled, y_test_scaled], axis=0)

    print(f"{data_splitting=}, {label_frac=}")
    print(f"X_L: {X_L.shape}, y_L: {y_L.shape}")
    print(f"X_U: {X_U.shape}, y_U: {y_U.shape}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"X_small: {X_small.shape}")

    timevae_file_path = None
    if params["run_console"]["timevae"] == True:
        timevae_file_name = f"X.npz"
        timevae_folder    = f"{interim_data_loc}/timevae/{desired_dataset}_frac{label_frac}"
        os.makedirs(timevae_folder, exist_ok=True)
        np.savez_compressed(f"{timevae_folder}/{timevae_file_name}", data=np.array(X_small, dtype=np.float32))
        print(f"Saved {timevae_file_name} to {timevae_folder}")
        timevae_file_path = f"{timevae_folder}/{timevae_file_name}"

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, X_small (TimeVAE): {X_small.shape}")
    print(f"y_train: {y_train_scaled.shape}, y_test: {y_test_scaled.shape}, y_small (TimeVAE): {y_small.shape}")
    print(f"Mean: {np.mean(X_train):.2f}, {np.mean(X_test):.2f}, {y_train_scaled.mean():.2f}, {y_test_scaled.mean():.2f}")

    return X_L, y_L, X_U, y_U, y_train_scaled, y_test_scaled, timevae_file_path
