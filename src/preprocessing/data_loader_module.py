import __main__
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

from preprocessing.dataset_preprocessors import DatasetPreprocessor, ECGLoader, NasaLoader, GermanyDataset, WeatherDataset, process_argoverse_parquet
from preprocessing.window_folder import WindowFolder
from param_config.config_paths import interim_data_loc, public_data_loc, encoders_folder, ts2vec_params_loc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetLoading:
    @staticmethod
    def load_ecg_data():
        ECG_data_path= "../public_datasets/3D/ptb-xl-1.0.3"
        loader       = ECGLoader(ECG_data_path)
        X, y, _, _   = loader.load_dataset(sampling="lr", target="diagnostic_superclass_multi",
                                        segment_duration_sec=200, max_records=2000,
                                        continuous_target=True)
        return X, y

    @staticmethod
    def load_argoverse_data():
        argoverse_data_path = "../public_datasets/3D/argoverse_forecasting"
        folder_name = "00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff"
        file_name   = "scenario_00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff.parquet"
        return process_argoverse_parquet(f"{argoverse_data_path}/{folder_name}/{file_name}")

    @staticmethod
    def load_asm_data():
        "For info on processing search term 'Fouad intervening', points to a cell in the ASM notebook"
        X_3d  = np.load("../public_datasets/3D/ASM/X_3d.npy")
        y_asm = np.load("../public_datasets/3D/ASM/y_asm.npy")
        return X_3d, y_asm

    @staticmethod
    def load_china_data() -> tuple[np.ndarray, np.ndarray]:
        """Load China weather, split first NUM_PAGES_TO_USE stations into SPLIT_RATIO windows."""
        dataset_location = "../public_datasets/3D/china_weather/weather2k.npy"
        china_data       = np.load(dataset_location, mmap_mode='r').transpose(0, 2, 1)
        print(f"Original China data shape: {china_data.shape}")

        y_indices = [4, 5, 6, 7, 10]
        mask      = np.ones(china_data.shape[2], dtype=bool)
        mask[y_indices] = False
        X_cut  = china_data[:, :, mask]         # (stations, timesteps, n_features)
        y_cut  = china_data[:, :, y_indices]    # (stations, timesteps, n_targets)

        print(f"X_cut: {X_cut.shape}, y_last: {y_cut.shape}")
        return X_cut, y_cut

    @staticmethod
    def load_gas_data() -> tuple[np.ndarray, np.ndarray]:
        """Load gas CSVs and produce X and full y per page, pad pages to max length
        Load gas CSVs and produce X and y windows of uniform length NUM_ROWS.
        Steps:
        1. Read all CSVs in the folder.
        2. Split each CSV into as many full windows of NUM_ROWS as possible.
        Extra rows that don't fit a window are discarded.
        3. For each window, take the last row as y.
        4. Stack all X windows and y rows, preserving order.
        5. If total number of windows > NUM_PAGES_TO_USE, truncate to NUM_PAGES_TO_USE.
        Returns:
            X_windows: (NUM_PAGES_TO_USE, NUM_ROWS, n_features)
            y_windows: (NUM_PAGES_TO_USE, n_targets)"""
        gas_folder = "../public_datasets/3D/gas_emissions"
        csv_files  = sorted([f for f in os.listdir(gas_folder) if f.endswith(".csv")])

        X_pages, y_pages = [], []

        # load all pages first
        for file in csv_files:
            df     = pd.read_csv(os.path.join(gas_folder, file))
            X_file = df.iloc[:, :-2].values      # features (timesteps × features)
            y_file = df.iloc[:, -2:].values      # targets  (timesteps × 2)

            # Outlier clipping
            for i in range(y_file.shape[1]):
                mean, std    = y_file[:, i].mean(), y_file[:, i].std()
                y_file[:, i] = np.clip(y_file[:, i], mean - 3 * std, mean + 3 * std)
            X_pages.append(X_file)
            y_pages.append(y_file)

        # pad pages to max length
        max_len_X = max(x.shape[0] for x in X_pages)
        max_len_y = max(y.shape[0] for y in y_pages)

        X_full = np.stack([np.pad(x, ((0, max_len_X - x.shape[0]), (0,0))) for x in X_pages], axis=0)
        y_full = np.stack([np.pad(y, ((0, max_len_y - y.shape[0]), (0,0))) for y in y_pages], axis=0)

        print(f"Gas dataset (padded): X={X_full.shape}, y={y_full.shape}")
        return X_full, y_full

    @staticmethod
    def load_germany_data() -> tuple[np.ndarray, np.ndarray]:
        """Load CAMELS-DE dataset, split each basin's timeseries into SPLIT_RATIO windows."""
        camels_root_folder = f"{public_data_loc}/3D/camels_de"
        timeseries_folder  = os.path.join(camels_root_folder, "timeseries")
        zarr_path          = os.path.join(camels_root_folder, "camels_de_timeseries.zarr")

        if os.path.exists(zarr_path):
            X = xr.open_zarr(zarr_path)["X"].values  # (basins, timesteps, features)
        else:
            X = GermanyDataset.load_X_from_scratch(timeseries_folder, zarr_path)
        y = GermanyDataset.load_y_from_scratch(camels_root_folder)  # (basins, targets)

        # ---
        total_nan_frac = np.isnan(X).sum() / X.size
        total_nan_frac_y = np.isnan(y).sum() / y.size
        print(f"X missing: {total_nan_frac:.3%}, y missing: {total_nan_frac_y:.3%}")
        # ---
        X, y = np.nan_to_num(X), np.nan_to_num(y)
        return X, y

    @staticmethod
    def load_india_data():
        """Load India catchment dataset, split into windows like WeatherBench."""
        data_path      = "../public_datasets/3D/india_catchments"
        forcing_folder = "catchment_mean_forcings"
        clim_file      = "attributes_csv/camels_ind_clim.csv"

        y_df           = pd.read_csv(f"{data_path}/{clim_file}")
        catchment_ids  = y_df.iloc[:, 0].astype(int).values
        y_all          = y_df.iloc[:, 1:]

        forcing_files    = sorted(os.listdir(f"{data_path}/{forcing_folder}"))
        X_list, file_ids = [], []
        for f in forcing_files:
            df = pd.read_csv(os.path.join(data_path, forcing_folder, f))
            X_list.append(df.drop(columns=['year','month','day','pet(mm/day)']).values)
            file_ids.append(int(f.split('.')[0]))

        X     = np.stack(X_list, axis=0)
        order = [file_ids.index(cid) for cid in catchment_ids]
        X     = X[order]
        y     = y_all.values

        le = LabelEncoder()
        for i in range(y.shape[1]):
            if isinstance(y[0, i], str):
                le.fit(y[:, i])
                y[:, i] = le.transform(y[:, i])
        y = y.astype(float)
        return X, y

    @staticmethod
    def load_nasa_data(nasa_folder="../public_datasets/3D/NASA", specific_file="FD004",
                       single_target: bool=False):
        """Load and preprocess NASA FD004 dataset, returning padded and scaled arrays."""

        # --- Load files ---
        train_file = f"{nasa_folder}/train_{specific_file}.txt"
        test_file  = f"{nasa_folder}/test_{specific_file}.txt"
        rul_file   = f"{nasa_folder}/RUL_{specific_file}.txt"

        cols       = ['unit', 'cycle'] + [f'op{i}' for i in range(1, 4)] + [f's{i}' for i in range(1, 22)]
        train_df   = pd.read_csv(train_file, delim_whitespace=True, header=None, names=cols).dropna(axis=1, how='all')
        test_df    = pd.read_csv(test_file,  delim_whitespace=True, header=None, names=cols).dropna(axis=1, how='all')
        rul_series = pd.read_csv(rul_file, header=None).iloc[:,0]

        # --- Compute RUL ---
        train_df['RUL'] = train_df.groupby('unit')['cycle'].transform(lambda x: x.max() - x)
        last_cycle = test_df.groupby('unit')['cycle'].max().to_dict()
        test_df['RUL'] = [rul_series[row['unit']-1] + (last_cycle[row['unit']] - row['cycle'])
                           for _, row in test_df.iterrows()]

        # --- Scale features ---
        feature_cols = cols[2:]
        scaler       = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

        # --- Fold by engine and pad ---
        X_train, y_train, _ = NasaLoader.fold_by_engine_unit(train_df, feature_cols, single_target=single_target)
        X_test,  y_test,  _ = NasaLoader.fold_by_engine_unit(test_df,  feature_cols, single_target=single_target)

        max_len = max(X_train.shape[1], X_test.shape[1])
        X_train = NasaLoader.pad_X_to_max(X_train, max_len)
        X_test  = NasaLoader.pad_X_to_max(X_test,  max_len)
        y_train = NasaLoader.pad_y_to_max(y_train, max_len)
        y_test  = NasaLoader.pad_y_to_max(y_test,  max_len)

        # --- Scale target ---
        y_scaler       = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled  = y_scaler.transform(y_test)
        return X_train, X_test, y_train_scaled, y_test_scaled

    @staticmethod
    def load_panama_data() -> tuple[np.ndarray, np.ndarray]:
        """Load Panama electricity, split first NUM_PAGES_TO_USE stations into SPLIT_RATIO windows."""
        file_name     = "train.csv"
        file_location = f"{public_data_loc}/3D/panama/{file_name}"

        df        = pd.read_csv(file_location)
        df        = df.drop(columns=['datetime'])

        y_indices = [1] # idx of valid y
        X_cut     = df.drop(df.columns[y_indices], axis=1).to_numpy()
        y_cut     = df.iloc[:, y_indices].to_numpy().reshape(-1, 1)  # make 2D

        X_cut = X_cut[None, :, :]
        y_cut = y_cut[None, :, :]
        return X_cut, y_cut

    @staticmethod
    def load_weather_data() -> tuple[np.ndarray, np.ndarray]:
        """Load WeatherBench data, keep the first NUM_PAGES_TO_USE spatial locations,
        split each series into SPLIT_RATIO contiguous windows, and truncate
        each window to NUM_ROWS timesteps. Returns ML-ready (X, y) arrays
        where y is the last timestep of each window.
        Returns:
            X_windows: (NUM_PAGES_TO_USE * SPLIT_RATIO, page_len, n_features)
            y_windows: (NUM_PAGES_TO_USE * SPLIT_RATIO, n_targets)"""
        variables_X    = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
        variables_y    = ["mean_sea_level_pressure", "total_precipitation_6hr"]
        dataset_folder = "../public_datasets/3D/weather_bench"
        weather        = WeatherDataset(dataset_folder, variables_X, variables_y)
        X_xarr, y_xarr = weather.load_dataset()

        X_3d           = weather.prepare_X_features(X_xarr, mode="3d")  # (time, channels, lat*lon)
        X_pages        = weather.reshape_for_ml(X_3d)

        y_time_series = np.stack([y_xarr[var].values for var in y_xarr.data_vars], axis=-1)
        time_dim, lat_dim, lon_dim, num_targets = y_time_series.shape
        y_long_series = y_time_series.reshape(time_dim, -1, num_targets)
        y_pages       = np.moveaxis(y_long_series, [0, 1, 2], [1, 0, 2])
        print(f"Original Weather shape: {X_pages.shape}, {y_pages.shape}")
        return X_pages, y_pages

    @staticmethod
    def load_beijing_data() -> tuple[np.ndarray, np.ndarray]:
        """Load Beijing Air Quality dataset, encode wind direction, handle missing values.
        - X: (stations, timesteps, features) unwindowed
        - y: (stations, timesteps, 1) unwindowed
        Folding, train/test split, and scaling are handled by load_or_preprocess_dataset."""
        data_path    = "../public_datasets/3D/beijing"
        target_col   = 'PM2.5'
        cols_to_drop = ['No','year','month','day','hour','station']

        def encode_wind_direction_simple(wd_series):
            wd_map = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,
                      'E':90,'ESE':112.5,'SE':135,'SSE':157.5,
                      'S':180,'SSW':202.5,'SW':225,'WSW':247.5,
                      'W':270,'WNW':292.5,'NW':315,'NNW':337.5}
            angles = wd_series.map(wd_map).fillna(0.0).values
            sin_wd = np.sin(np.deg2rad(angles))
            cos_wd = np.cos(np.deg2rad(angles))
            return np.stack([sin_wd, cos_wd], axis=-1)

        X_list, y_list = [], []
        for file in sorted(os.listdir(data_path)):
            if not file.endswith(".csv"):
                continue
            df = pd.read_csv(f"{data_path}/{file}")
            df = df.drop(columns=cols_to_drop, errors='ignore')

            # Encode wind direction
            if 'wd' in df.columns:
                wd_encoded = encode_wind_direction_simple(df['wd'])
                df         = df.drop(columns=['wd'])
                df[['wd_sin','wd_cos']] = wd_encoded

            # Separate features and target
            feature_cols = [c for c in df.columns if c != target_col]
            X_station    = df[feature_cols].values.astype(np.float32)
            y_station    = df[[target_col]].values.astype(np.float32)  # 2D: (timesteps, 1)
            X_list.append(X_station)
            y_list.append(y_station)

        X_all = np.stack(X_list, axis=0)  # (stations, timesteps, features)
        y_all = np.stack(y_list, axis=0)  # (stations, timesteps, 1)

        # apply per–station masking
        isnan_mask       = ~np.isnan(y_all[..., 0])
        X_clean, y_clean = [], []
        for s in range(X_all.shape[0]):
            m = isnan_mask[s]
            X_clean.append(X_all[s][m])
            y_clean.append(y_all[s][m])

        min_T = min(x.shape[0] for x in X_clean)
        X_all = np.stack([x[:min_T] for x in X_clean], axis=0)
        y_all = np.stack([y[:min_T] for y in y_clean], axis=0)
        print(f"after dropping missing y: X={X_all.shape}, y={y_all.shape}")

        total_nan_frac_X = np.isnan(X_all).sum() / X_all.size
        total_nan_frac_y = np.isnan(y_all).sum() / y_all.size

        print(f"Beijing X missing: {total_nan_frac_X:.3%}, y missing: {total_nan_frac_y:.3%}")
        X_all = np.nan_to_num(X_all)
        y_all = np.nan_to_num(y_all)

        # plt.figure(figsize=(14, 14))
        # # plt.plot(X_all[1, :2_100, 0])
        # plt.scatter(np.arange(X_all.shape[1]), X_all[0, :, 0], s=1)
        # plt.title("Beijing Air Quality - PM2.5")
        # plt.xlabel("Time")
        # plt.ylabel("PM2.5 Concentration")
        # plt.show()
        print(f"Beijing dataset loaded: X={X_all.shape}, y={y_all.shape}")
        return X_all, y_all


def load_or_preprocess_dataset(desired_dataset: str, page_num: int, do_we_scale_y: bool, dataset_window=None, 
                               random_seed=None, use_cache=False, num_rows_per_window=None) -> tuple[np.ndarray, ...]:
    """Load cached preprocessed dataset if available, otherwise preprocess and cache it."""
    new_dir_name   = f"{desired_dataset}_{page_num}pages"
    save_dir       = f"{interim_data_loc}/{new_dir_name}"
    X_full, y_full = dataset_loaders_dict[desired_dataset]()
    print(f"X_train: {X_full.shape} ({X_full.nbytes/1024**2:.1f} MB)")
    print(f"y_full:  {y_full.shape} ({y_full.nbytes/1024**2:.1f} MB)")

    if desired_dataset == 'asm':
        return X_train, X_test, y_train_scaled, y_test_scaled, window_size

    if dataset_window is not None:
        X_full, y_full, window_size, _ = WindowFolder.auto_fold_timeseries(X_full, y_full, fs=1.0, peak_strength=2.0, fallback_window=num_rows_per_window, 
                    denoise=False, window_size=dataset_window, max_pages=page_num, num_pages_to_use=page_num)
        print(f"Using predefined dataset window size: {dataset_window}")
    else:
        X_full, y_full, window_size, _ = WindowFolder.auto_fold_timeseries(X_full, y_full, fs=1.0, peak_strength=2.0, fallback_window=num_rows_per_window,
                    denoise=False, window_size=None, max_pages=page_num, num_pages_to_use=page_num)
        print(f"Auto-detected dataset window size: {window_size}")
    print("Folded the dataset")

    if use_cache and os.path.exists(save_dir): # Load from cache
        print("Path exists, loading from cache:", save_dir)
        X_train        = np.load(f"{save_dir}/X_train.npz")['data']
        X_test         = np.load(f"{save_dir}/X_test.npz")['data']
        y_train_scaled = np.load(f"{save_dir}/y_train.npz")['data']
        y_test_scaled  = np.load(f"{save_dir}/y_test.npz")['data']

    else: # Preprocess fresh
        print("Path does not exist, preprocessing fresh:", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        preprocessor = DatasetPreprocessor(page_num=page_num, test_size=0.2, random_seed=random_seed)
        X_train, X_test, y_train_scaled, y_test_scaled = preprocessor.fit_transform(X_full, y_full, scale_y=do_we_scale_y)

        del X_full, y_full, preprocessor
        gc.collect()

        # Save to cache
        np.savez_compressed(f"{save_dir}/X_train.npz", data=X_train.astype(np.float32))
        np.savez_compressed(f"{save_dir}/X_test.npz", data=X_test.astype(np.float32))
        np.savez_compressed(f"{save_dir}/y_train.npz", data=y_train_scaled.astype(np.float32))
        np.savez_compressed(f"{save_dir}/y_test.npz", data=y_test_scaled.astype(np.float32))
    return X_train, X_test, y_train_scaled, y_test_scaled, window_size


dataset_loaders_dict = {
    "ecg":               DatasetLoading.load_ecg_data,
    "argoverse":         DatasetLoading.load_argoverse_data,
    "weather":           DatasetLoading.load_weather_data,
    "india_catchment":   DatasetLoading.load_india_data,
    "germany_catchment": DatasetLoading.load_germany_data,
    "china_weather":     DatasetLoading.load_china_data,
    "gas":               DatasetLoading.load_gas_data,
    "panama":            DatasetLoading.load_panama_data,
    "beijing":           DatasetLoading.load_beijing_data,
    "asm":               DatasetLoading.load_asm_data}
