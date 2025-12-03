import __main__
import os
from typing import Dict, List, Literal, Tuple, Optional
import numpy as np

from src.utils.data_utils import select_top_X_features
from src.preprocessing.data_loader_module import DatasetLoading, load_or_preprocess_dataset
import src.param_config.config_paths as P

# STEP 1: Load and preprocess the data
def load_the_data(desired_dataset: str, NUM_PAGES_TO_USE: int, do_we_scale_y: bool, dataset_window: int,
                  rand_seed: int, NUM_ROWS: int, params: dict, label_frac: float = None, use_cache=False):
    """[RUN ME] Preprocess dataset, as class"""

    if desired_dataset == "nasa": # Directly load NASA dataset
        X_train, X_test, y_train_scaled, y_test_scaled = DatasetLoading.load_nasa_data()
        window_size = "N/A"
    elif desired_dataset != "asm":
        X_train, X_test, y_train_scaled, y_test_scaled, window_size = load_or_preprocess_dataset(desired_dataset, NUM_PAGES_TO_USE, do_we_scale_y,
                    dataset_window=dataset_window, random_seed=rand_seed, use_cache=False, num_rows_per_window=NUM_ROWS)

    elif desired_dataset == "asm":
        from asm_stuff.main_runner import prepare_asm_train_test
        top_idx = [479, 517, 54, 165, 121, 77, 177, 188, 55, 509, 53, 52, 76,
                388, 125, 131, 84, 140, 81, 124, 189, 185, 206, 385, 160, 182,
                178, 75, 145, 144, 100, 306, 102, 205, 0, 149, 117, 163, 312,
                204, 98, 101, 157, 128, 207, 202, 159, 409, 158, 156, 415, 99,
                203, 103, 97, 201, 96, 200, 1]

        X_train, X_test, y_train_scaled, y_test_scaled = prepare_asm_train_test(P.asm_folder_loc, top_idx, keep_frac=0.08, keep='first')
        print(X_train.shape, y_train_scaled.shape)
        window_size = window_size if 'window_size' in locals() else X_train.shape[1]
        label_frac  = label_frac if 'label_frac' in locals() else 1

    if X_train.shape[2] > 300:
        top_features_pct         = params["general_params"]["top_features_big_dataset_pct"] # % top features to select
        X_train, X_test, top_idx = select_top_X_features(X_train, y_train_scaled, X_test, top_features_pct)
    elif X_train.shape[2] > 30:
        top_features_pct         = params["general_params"]["top_features_pct"] # % of top features to select
        X_train, X_test, top_idx = select_top_X_features(X_train, y_train_scaled, X_test, top_features_pct)

    # save the model here

    print(f"X_train: {X_train.shape} ({X_train.nbytes/1024**2:.1f} MB), X_test: {X_test.shape} ({X_test.nbytes/1024**2:.1f} MB)")
    print(f"y_train: {y_train_scaled.shape} ({y_train_scaled.nbytes/1024**2:.1f} MB), y_test: {y_test_scaled.shape} ({y_test_scaled.nbytes/1024**2:.1f} MB)")
    print(f"Mean: {X_train.mean():.2f}, {X_test.mean():.2f}, {y_train_scaled.mean():.2f}, {y_test_scaled.mean():.2f}")
    print(f"Y is scaled: {do_we_scale_y}")
    print("Min:", np.min(X_train), np.min(X_test))
    print("Max:", np.max(X_train), np.max(X_test))
    return X_train, X_test, y_train_scaled, y_test_scaled, window_size

# STEP 2: Split data into labeled + unlabeled portions
def split_data_to_labeled_unlabeled(desired_dataset, interim_data_loc, data_splitting, label_frac, X_train, y_train_scaled,
                                    X_test, y_test_scaled, params, rand_seed: int = None):
    """Given a data splitting method and its %, split the data into labeled and unlabeled portions."""
    n_train   = len(X_train)
    n_labeled = int(np.ceil(label_frac * n_train))

    if label_frac == 1.0:
        # If we use all labels, use the pre-existing order from the loader (OLD code behavior).
        shuffled_idx = np.arange(n_train) 
    elif rand_seed is not None:
        # Only perform the shuffle/permutation if we are actually subsampling (label_frac < 1.0)
        rng          = np.random.default_rng(rand_seed)
        shuffled_idx = rng.permutation(n_train)
    else:
        shuffled_idx = np.random.permutation(n_train)

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
        # timevae_file_name = f"X.npz"
        # timevae_folder    = f"{interim_data_loc}/timevae/{desired_dataset}_frac{label_frac}"
        # os.makedirs(timevae_folder, exist_ok=True)
        # np.savez_compressed(f"{timevae_folder}/{timevae_file_name}", data=np.array(X_small, dtype=np.float32))

        timevae_file_name = "TimeVAE_parameters.npz"
        timevae_folder    = f"{interim_data_loc}/timevae/{desired_dataset}_frac{label_frac}"
        os.makedirs(timevae_folder, exist_ok=True)
        np.savez_compressed(f"{timevae_folder}/{timevae_file_name}", data=np.array(X_small, dtype=np.float32))

        print(f"Saved {timevae_file_name} to {timevae_folder}")
        timevae_file_path = f"{timevae_folder}/{timevae_file_name}"

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, X_small (TimeVAE): {X_small.shape}")
    print(f"y_train: {y_train_scaled.shape}, y_test: {y_test_scaled.shape}, y_small (TimeVAE): {y_small.shape}")
    print(f"Mean: {np.mean(X_train):.2f}, {np.mean(X_test):.2f}, {y_train_scaled.mean():.2f}, {y_test_scaled.mean():.2f}")

    return X_L, y_L, X_U, y_U, y_train_scaled, y_test_scaled, timevae_file_path

