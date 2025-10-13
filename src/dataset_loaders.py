"""Downloads 3D datasets"""

from __future__ import annotations
import ast
import json
import os
import shutil
import wfdb

from glob import glob
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr

import category_encoders as ce
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetPreprocessor:
    """Preprocess datasets: downsample, train/test split, categorical encoding, and scaling."""

    def __init__(self, page_frac=0.1, row_frac=0.4, test_size=0.2, scale_X=True, random_seed=None):
        self.page_frac  = page_frac
        self.row_frac   = row_frac
        self.test_size  = test_size
        self.scale_X    = scale_X
        self.random_seed= random_seed
        self.y_mean     = None
        self.y_std      = None
        self.X_scaler   = None

    def _prepare_targets(self, y):
        if isinstance(y, np.ndarray):
            self.y_df         = pd.DataFrame(y)
            self.cat_cols     = []
            self.numeric_cols = self.y_df.columns.tolist()
        else:
            self.y_df         = y.copy()
            self.numeric_cols = self.y_df.select_dtypes(include=[np.number]).columns.tolist()
            self.cat_cols     = self.y_df.select_dtypes(include=['object','category']).columns.tolist()

    def _downsample_pages_and_rows(self, X, y_df):
        n_pages, n_rows, _ = X.shape
        num_pages = max(1, int(n_pages * self.page_frac))
        num_rows  = max(1, int(n_rows * self.row_frac))
        # page_idx  = np.linspace(0, n_pages-1, num_pages, dtype=int)
        # row_idx   = np.linspace(0, n_rows-1, num_rows, dtype=int)
        # X_small   = X[page_idx][:, row_idx, :]
        # y_small   = y_df.iloc[page_idx].values if isinstance(y_df, pd.DataFrame) else y_df[page_idx]
        # Randomly choose pages/rows instead of linspace
        rng = np.random.default_rng(self.random_seed)
        page_idx = rng.choice(n_pages, size=num_pages, replace=False)
        row_idx  = rng.choice(n_rows, size=num_rows, replace=False)
        
        X_small = X[page_idx][:, row_idx, :]
        y_small = y_df.iloc[page_idx].values if isinstance(y_df, pd.DataFrame) else y_df[page_idx]
        return X_small, y_small

    def _encode_categorical(self, y_train, y_test):
        if len(self.cat_cols) > 0:
            idx         = [self.y_df.columns.get_loc(c) for c in self.cat_cols]
            y_train_df  = pd.DataFrame(y_train[:, idx], columns=self.cat_cols)
            y_test_df   = pd.DataFrame(y_test[:, idx], columns=self.cat_cols)
            encoder     = ce.TargetEncoder(cols=self.cat_cols)
            y_train_enc = encoder.fit_transform(y_train_df, y_train[:,0])
            y_test_enc  = encoder.transform(y_test_df)
            y_train[:, idx] = y_train_enc.values
            y_test[:, idx]  = y_test_enc.values
        return y_train.astype(float), y_test.astype(float)

    def _scale_targets(self, y_train, y_test):
        self.y_mean    = y_train.mean(axis=0)
        self.y_std     = y_train.std(axis=0)
        self.y_std[self.y_std == 0] = 1.0
        y_train_scaled = (y_train - self.y_mean) / self.y_std
        y_test_scaled  = (y_test - self.y_mean) / self.y_std
        y_train_scaled = np.atleast_2d(y_train_scaled)
        y_test_scaled  = np.atleast_2d(y_test_scaled)
        return y_train_scaled, y_test_scaled

    # [old]
    def X_fit_transform(self, X: np.ndarray, y: np.ndarray, split_X: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Downsample, split, encode, and scale dataset. X always stays 3D."""
        self._prepare_targets(y)
        X_small, y_small = self._downsample_pages_and_rows(X, self.y_df)

        n_samples = X_small.shape[0]
        n_test    = int(n_samples * self.test_size)

        # 1. Shuffle samples then traintest split (based on random seed)
        rng           = np.random.default_rng(self.random_seed)
        indices       = rng.permutation(n_samples)
        train_indices = indices[n_test:]
        test_indices  = indices[:n_test]

        # 3. Split X and Y simultaneously using the shared indices
        y_train_raw, y_test_raw = y_small[train_indices], y_small[test_indices]
        X_train, X_test         = X_small[train_indices], X_small[test_indices]
        y_train, y_test         = self._encode_categorical(y_train_raw, y_test_raw)
        y_train_scaled, y_test_scaled = self._scale_targets(y_train, y_test)
        # =======
        # y_train, y_test  = train_test_split(y_small, test_size=self.test_size, shuffle=False)
        # y_train, y_test  = self._encode_categorical(y_train, y_test)
        # y_train_scaled, y_test_scaled = self._scale_targets(y_train, y_test)

        if not split_X:
            return X_small, y_train_scaled, y_test_scaled, None

        # Split X along first axis (pages) without flattening
        # X_train, X_test = train_test_split(X_small, test_size=self.test_size, shuffle=False)

        if self.scale_X:
            # Optionally scale while keeping 3D shape
            ns, nr, nf = X_train.shape
            ns_test, nr_test, nf_test = X_test.shape

            # Flatten temporarily for StandardScaler
            X_train_flat   = X_train.reshape(ns, -1)
            X_test_flat    = X_test.reshape(ns_test, -1)
            self.X_scaler  = StandardScaler()
            X_train_scaled = self.X_scaler.fit_transform(X_train_flat).reshape(ns, nr, nf)
            X_test_scaled  = self.X_scaler.transform(X_test_flat).reshape(ns_test, nr_test, nf_test)
            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
        else:
            return X_train, X_test, y_train_scaled, y_test_scaled

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Downsample, split, encode, and scale dataset. X always stays 3D."""
        self._prepare_targets(y)
        X_small, y_small = self._downsample_pages_and_rows(X, self.y_df)

        n_samples = X_small.shape[0]
        n_test    = int(n_samples * self.test_size)

        # 1. Shuffle samples then traintest split (based on random seed)
        rng           = np.random.default_rng(self.random_seed)
        indices       = rng.permutation(n_samples)
        train_indices = indices[n_test:]
        test_indices  = indices[:n_test]

        # 3. Split X and Y simultaneously using the shared indices
        y_train_raw, y_test_raw = y_small[train_indices], y_small[test_indices]
        X_train, X_test         = X_small[train_indices], X_small[test_indices]
        y_train, y_test         = self._encode_categorical(y_train_raw, y_test_raw)
        y_train_scaled, y_test_scaled = self._scale_targets(y_train, y_test)

        if self.scale_X:
            ns, nr, nf = X_train.shape
            ns_test, nr_test, nf_test = X_test.shape

            # Flatten temporarily for StandardScaler
            X_train_flat   = X_train.reshape(ns, -1)
            X_test_flat    = X_test.reshape(ns_test, -1)
            self.X_scaler  = StandardScaler()
            X_train_scaled = self.X_scaler.fit_transform(X_train_flat).reshape(ns, nr, nf)
            X_test_scaled  = self.X_scaler.transform(X_test_flat).reshape(ns_test, nr_test, nf_test)
            return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
        else:
            return X_train, X_test, y_train_scaled, y_test_scaled



class ECGLoader:
    """Load + process the 'PTB-XL' ECG dataset (https://physionet.org/content/ptb-xl/1.0.3/), including SCP code parsing and label encoding
    Attributes:
        X: 3D array of shape (num_records, num_samples, num_leads), ECG signals.
        y: 2D array of labels, either multi-hot (multi-class) or code confidences (0-100).
        fs: Sampling frequency of the signals (Hz).
        lead_names: List of ECG channel names
    Parameters:
        base_dir: Path to PTB-XL dataset.
        sampling: 'hr' for high-resolution (5000 samples/10s), 'lr' for low-resolution (1000 samples/10s).
        target: 'multi' for multi-hot labels, 'single' for majority-superclass integer labels.
        segment_duration_sec: Desired segment length in seconds; signals are cropped or zero-padded.
        max_records: Optional limit on number of records to load.
        continuous_target: If True, returns code confidence values (0-100) instead of binary labels."""

    def __init__(self, base_dir: str | Path):
        self.base = Path(base_dir)

    @staticmethod
    def parse_scp_codes(s: str) -> dict[str, float]:
        """Parse SCP code string into a dictionary of code → confidence (0-100)."""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return ast.literal_eval(s)

    @staticmethod
    def extract_superclasses(scp_codes_str: str, scp_super) -> List[str]:
        """Return list of diagnostic conditions 'superclasses' present in SCP string."""
        d            = ECGLoader.parse_scp_codes(scp_codes_str)
        present_scps = [k for k, v in d.items() if float(v) > 0]
        supers       = [scp_super[s] for s in present_scps if s in scp_super]
        return sorted(set(supers))

    @staticmethod
    def compute_majority_superclass(scp_codes_str: str, scp_super) -> str | None:
        """Return the diagnostic conditions 'superclass' with the highest summed confidence."""
        d = ECGLoader.parse_scp_codes(scp_codes_str)
        agg: Dict[str, float] = {}
        for k, w in d.items():
            if k in scp_super:
                agg[scp_super[k]] = agg.get(scp_super[k], 0.0) + float(w)
        return max(agg.items(), key=lambda kv: kv[1])[0] if agg else None

    @staticmethod
    def encode_scp_vector(scp_str: str, all_codes: list[str]) -> np.ndarray:
        """Convert SCP string into a vector of code confidences for all_codes."""
        d = ECGLoader.parse_scp_codes(scp_str)
        return np.array([d.get(code, 0.0) for code in all_codes], dtype=np.float32)

    def load_dataset(self, sampling: Literal["hr", "lr"] = "lr",
                     target: Literal["diagnostic_superclass_multi", "diagnostic_superclass_single"] = "diagnostic_superclass_multi",
                     segment_duration_sec: float | None = 10.0, max_records: int | None = None,
                     continuous_target: bool=False) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
        """Load PTB-XL ECGs.
        - sampling: "hr" = high-res signal (500hz), "lr" = low-res signal (100Hz)
        - target: label (y) format. "multi" = vector of values, "single" = 1 value
        - segment_duration_sec: desired duration of each ECG segment in s (up to the full record length, typically 10s);
        signals are truncated if longer or zero-padded if shorter to produce a uniform number of samples per segment
        - max_records: max # of records to load (otherwise it becomes too big)
        - continuous_target: if True, we get y labels as raw confidences (0-100) instead of binary values
        Returns X (signals), y (labels), sampling_rate, leads."""
        meta      = pd.read_csv(self.base / "ptbxl_database.csv")
        scp       = pd.read_csv(self.base / "scp_statements.csv", index_col=0)
        scp_diag  = scp[scp["diagnostic"] == 1].index.tolist()
        fname_col = "filename_hr" if sampling == "hr" else "filename_lr"
        meta      = meta[[fname_col, "scp_codes"]].copy()

        if max_records is not None:
            meta = meta.iloc[:max_records]

        scp_super    = scp.loc[scp_diag, "diagnostic_class"].to_dict()
        classes      = sorted(set(scp_super.values()))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        rec_supers: List[List[str]] = meta["scp_codes"].map(lambda s: ECGLoader.extract_superclasses(s, scp_super)).tolist()

        if continuous_target:
            all_codes = sorted({code for scp_str in meta["scp_codes"] for code in ECGLoader.parse_scp_codes(scp_str)})
            y         = np.stack([ECGLoader.encode_scp_vector(s, all_codes) for s in meta["scp_codes"]], axis=0)
        elif target == "diagnostic_superclass_multi":
            y = np.zeros((len(rec_supers), len(classes)), dtype=np.float32)
            for i, supers in enumerate(rec_supers):
                for s in supers:
                    y[i, class_to_idx[s]] = 1.0
        else:
            majors = meta["scp_codes"].map(lambda s: ECGLoader.compute_majority_superclass(s, scp_super)).tolist()
            y      = np.array([class_to_idx[m] if m is not None else -1 for m in majors], dtype=np.int64)

        sample_path   = self.base / meta.iloc[0][fname_col]
        sig0, fields0 = wfdb.rdsamp(str(sample_path))
        sampling_rate = int(fields0["fs"])
        leads         = fields0["sig_name"]
        target_len    = int(sampling_rate * segment_duration_sec) if segment_duration_sec is not None else None

        X_list: List[np.ndarray] = []
        for p in meta[fname_col].tolist():
            sig, _ = wfdb.rdsamp(str(self.base / p))
            sig    = sig.astype(np.float32)
            if segment_duration_sec is not None:
                T = sig.shape[0]
                if T >= target_len:
                    sig = sig[:target_len, :]
                else:
                    sig = np.pad(sig, ((0, target_len - T), (0, 0)), mode="constant")
            X_list.append(sig)
        X = np.stack(X_list, axis=0)
        return X, y, sampling_rate, leads


class GermanyDataset:
    @staticmethod
    def load_X_from_scratch(timeseries_folder, zarr_path):
        # --- Delete Zarr if needed ---
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)

        # --- Load X ---
        ts_files  = sorted(glob(os.path.join(timeseries_folder, "*.csv")))
        ts_arrays = []

        for f in ts_files:
            df  = pd.read_csv(f, index_col=0)  # (time, catchments)
            df  = df.drop(columns=['date', 'discharge_vol_obs', 'discharge_spec_obs', 'water_level_obs'])
            arr = df.values
            ts_arrays.append(arr)

        # Stack along new axis -> (time, catchments, variables)
        X_np = np.stack(ts_arrays, axis=2)
        X_np = np.transpose(X_np, (2, 0, 1))
        print("X shape after transpose:", X_np.shape)  # (1582, 25568, 21)

        X_da = xr.DataArray(X_np, dims=("catchment", "time", "variable"))
        X_da.to_dataset(name="X").to_zarr(zarr_path, mode="w")
        print("Saved X to Zarr:", zarr_path)

        X_da_lazy = xr.open_zarr(zarr_path)["X"].values
        return X_da_lazy

    @staticmethod
    def load_y_from_scratch(attributes_folder):
        """Also removed str cols"""
        attr_files = glob(os.path.join(attributes_folder, "CAMELS_DE_*.csv"))
        y_list     = [pd.read_csv(f, index_col=0) for f in attr_files]
        y_germany  = pd.concat(y_list, axis=1)
        y_germany = y_germany.select_dtypes(exclude='object').to_numpy()
        return y_germany


class WeatherDataset:
    """Loads and preprocesses WeatherBench-style datasets for ML.
    Supports multiple output shapes for X and y"""
    def __init__(self, dataset_folder: str, variables_X: list, variables_y: list, freq="6H"):
        self.dataset_folder = dataset_folder
        self.variables_X    = variables_X
        self.variables_y    = variables_y
        self.freq   = freq
        self.X_xarr = None
        self.y_xarr = None

    @staticmethod
    def open_zarr_variable(folder_path: str, varname: str) -> xr.DataArray:
        """Open a folder containing a single variable as xarray.DataArray"""
        arr = zarr.open(folder_path, mode="r")
        time= pd.date_range("1959-01-01", periods=arr.shape[0], freq="6H")
        lat = np.linspace(-90, 90, arr.shape[1])
        lon = np.linspace(0, 360, arr.shape[2], endpoint=False)
        return xr.DataArray(arr, dims=["time", "lat", "lon"], coords={"time": time, "lat": lat, "lon": lon}, name=varname)

    def load_dataset(self):
        """Load all variables into xarray Datasets for X and y"""
        X_data      = {var: self.open_zarr_variable(f"{self.dataset_folder}/{var}", var) for var in self.variables_X}
        y_data      = {var: self.open_zarr_variable(f"{self.dataset_folder}/{var}", var) for var in self.variables_y}
        self.X_xarr = xr.Dataset(X_data)
        self.y_xarr = xr.Dataset(y_data)
        return self.X_xarr, self.y_xarr

    @staticmethod
    def prepare_X_features(X_ds: xr.Dataset, mode: str = "autoencoder", last_timesteps: int = 1) -> np.ndarray:
        """Convert xarray.Dataset of features to NumPy array
        - "autoencoder" [for autoencoder]  -> keep channels and spatial dims (time x channels x lat x lon)
        - "3d" [for 3D CNN]                -> keep channels, flatten spatial dims (time x channels x lat*lon)
        - "lstm" [for LSTM/Transformer]    -> flatten spatial dims, keep time (time x features)
        - "catboost" [for CatBoost]        -> flatten channels and spatial dims for last timestep only (1 x features)"""
        X_arr = np.stack([X_ds[var].values for var in X_ds.data_vars], axis=1)  # (time, channels, lat, lon)
        if mode == "autoencoder":
            return X_arr
        elif mode == "3d":
            return X_arr.reshape(X_arr.shape[0], X_arr.shape[1], -1)  # (time, channels, lat*lon)
        elif mode == "lstm":
            time_dim, channels, lat, lon = X_arr.shape
            return X_arr.reshape(time_dim, channels * lat * lon)
        elif mode == "catboost":
            X_last = X_arr[-last_timesteps:]
            return X_last.reshape(last_timesteps, -1)
        else:
            raise ValueError("mode must be one of ['autoencoder','3d','lstm','catboost']")

    @staticmethod
    def prepare_y_targets(y_ds: xr.Dataset, mode: str = "3d") -> np.ndarray:
        """Convert xarray.Dataset of targets to NumPy array"""
        y_arr  = np.stack([y_ds[var].values for var in y_ds.data_vars], axis=0)  # (vars, time, lat, lon)
        y_last = y_arr[:, -1, :, :]  # take last timestep
        if mode == "3d":
            return y_last  # (vars, lat, lon)
        elif mode == "flatten":
            return y_last.reshape(1, -1)  # (1, vars*lat*lon)
        elif mode == "collapse":
            return y_last.mean(axis=1)  # (vars, lon)
        else:
            raise ValueError("mode must be one of ['3d','flatten','collapse']")

    @staticmethod
    def reshape_for_ml(X: np.ndarray) -> np.ndarray:
        """Convert X from (time, channels, lat*lon) -> (lat*lon, time, channels)
        for per-grid-cell sequence models like timeVAE."""
        return np.moveaxis(X, [0, 1, 2], [1, 2, 0])

    @staticmethod
    def flatten_y(y: np.ndarray) -> np.ndarray:
        """Flatten y from (vars, lat, lon) -> (lat*lon, vars)
        to align with reshaped X."""
        return y.reshape(y.shape[0], -1).T

