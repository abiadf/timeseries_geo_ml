import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def _load_asm_data(folder_loc, top_idx=None, keep_frac=0.1, keep='last', mmap=True):
    """Load ASM data efficiently, trim to top feature indices, and keep only first/last portion of timesteps.
    Args:
        top_idx (list[int] | None): feature indices to keep.
        keep_frac (float): fraction of timesteps to retain (0 < keep_frac <= 1).
        keep (str): 'first' or 'last' to choose which segment of timesteps to keep.
        mmap (bool): use memory mapping to avoid loading whole array into RAM."""
    load_mode = 'r' if mmap else None
    X_3d  = np.load(f"{folder_loc}/X_3d.npy", mmap_mode=load_mode)
    y_asm = np.load(f"{folder_loc}/y_asm.npy", mmap_mode=load_mode)

    print(f"Loaded ASM data: X_3d shape {X_3d.shape}, y_asm shape {y_asm.shape}")

    # keep subset of features
    if top_idx is not None:
        X_3d = X_3d[:, :, top_idx]

    # select fraction of timesteps
    n_timesteps = X_3d.shape[1]
    keep_len = int(n_timesteps * keep_frac)
    if keep == 'last':
        X_3d = X_3d[:, -keep_len:, :]
    elif keep == 'first':
        X_3d = X_3d[:, :keep_len, :]
    else:
        raise ValueError("keep must be 'first' or 'last'")

    print(f"X_3d trimmed to shape {X_3d.shape} ({keep_frac*100:.1f}% timesteps kept)")
    print(f"y_asm shape remains {y_asm.shape}")
    return X_3d, y_asm

def prepare_asm_train_test(folder_loc, top_idx, keep_frac=0.1, keep='last', mmap=True, test_size=0.2):
    """Full pipeline: load, trim, split, scale."""
    X_3d, y_asm = _load_asm_data(folder_loc, top_idx=top_idx, keep_frac=keep_frac, keep=keep, mmap=mmap)
    X_train, X_test, y_train, y_test = train_test_split(X_3d, y_asm, test_size=test_size, random_state=42)

    scaler       = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat  = X_test.reshape(-1, X_test.shape[-1])
    scaler.fit(X_train_flat)

    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled  = scaler.transform(X_test_flat).reshape(X_test.shape)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled  = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train_scaled, X_test_scaled, y_train, y_test

top_idx = [479, 517,  54, 165, 121,  77, 177, 188,  55, 509,  53,  52,  76,
           388, 125, 131,  84, 140,  81, 124, 189, 185, 206, 385, 160, 182,
           178,  75, 145, 144, 100, 306, 102, 205,   0, 149, 117, 163, 312,
           204,  98, 101, 157, 128, 207, 202, 159, 409, 158, 156, 415,  99,
           203, 103,  97, 201,  96, 200,   1]
