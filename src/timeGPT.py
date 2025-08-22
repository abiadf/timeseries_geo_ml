"""Module running the TimeGPT baseline, which forecasts y"""

import math
import logging
from typing import Literal, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import PchipInterpolator

from nixtla import NixtlaClient # timeGPT

class TimeGPTForecaster:
    """Forecast multiple targets using TimeGPT with optional preprocessing."""
    def __init__(self, df: pd.DataFrame, time_col: str, y_cols: list, api_key: str,
                 model: str = "timegpt-1-long-horizon"):
        self.df       = df.copy()
        self.time_col = time_col
        self.y_cols   = y_cols
        self.model    = model
        self.client   = NixtlaClient(api_key=api_key)
        
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        self.X_cols        = None
        self.forecast_dict = {}
        self.forecast_dfs  = {} 

    @staticmethod
    def infer_dominant_freq(df: pd.DataFrame, time_col: str) -> str:
        """Return the most common time difference as pandas freq string."""
        sorted_df = df.sort_values(time_col)
        td        = sorted_df[time_col].diff().dropna()
        dominant_delta = td.mode()[0]
        return pd.tseries.frequencies.to_offset(dominant_delta).freqstr

    @staticmethod
    def apply_pchip_interpolation(df: pd.DataFrame, time_col: str, freq: str) -> pd.DataFrame:
        """Apply PCHIP interpolation to numeric columns on uniform time grid."""
        df            = df.copy()
        df[time_col]  = pd.to_datetime(df[time_col])
        df            = df.set_index(time_col).sort_index()
        uniform_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        x             = (df.index - df.index[0]).total_seconds()
        x_new         = (uniform_index - df.index[0]).total_seconds()
        resampled     = pd.DataFrame(index=uniform_index)
        for col in df.select_dtypes(include="number").columns:
            y             = df[col].values
            interp        = PchipInterpolator(x, y, extrapolate=False)
            resampled[col]= interp(x_new)
        return resampled.reset_index().rename(columns={"index": time_col})

    @staticmethod
    def compute_horizon(series_length: int, fixed_points: int, pct: float) -> int:
        """Compute forecast horizon as min of fixed points or percentage of series."""
        num_steps = max(fixed_points, math.ceil(series_length * pct))
        print(f"Horizon = {num_steps} steps (fixed_points={fixed_points}, pct={pct})")
        return num_steps

    def make_windows(self, n_windows = 10, horizon_len=None, horizon_frac=0.01, min_length = 300, start_point= 0):
        """Create rolling-origin windows for a multivariate df.
        df: pd.DataFrame, shape [T, n_features]
        n_windows: number of expanding windows
        horizon_frac: % of ENTIRE series length to use as forecast horizon
        Returns: list of tuples (train_df, test_df)
        min_length: min length of training set, mandated by nixtla/TimeGPT to 300"""
        if horizon_len is None:
            horizon_len = max(1, int(len(self.df) * horizon_frac))
        print(f"Horizon = {horizon_len} steps")
        max_train_len = len(self.df) - horizon_len
        window_sizes  = np.linspace(min_length, max_train_len, n_windows, dtype=int)
        windows_list  = []

        for w in window_sizes:
            train_df = self.df.iloc[start_point:w].copy()
            test_df  = self.df.iloc[w : w + horizon_len].copy()
            windows_list.append((train_df, test_df))
        return windows_list

    def _prepare_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame, horizon_len):
        """Scale features and split into train/test sets."""
        if self.X_cols is None:
            self.X_cols = [c for c in df_train.columns if c not in [self.time_col] + self.y_cols]

        df_train_scaled              = df_train.copy()
        df_train_scaled[self.X_cols] = self.scaler_X.fit_transform(df_train[self.X_cols])
        df_train_scaled[self.y_cols] = self.scaler_Y.fit_transform(df_train[self.y_cols])
        
        df_test = df_test.iloc[:horizon_len].copy()

        X_future_scaled = pd.DataFrame(self.scaler_X.transform(df_test[self.X_cols]),
                                       columns=self.X_cols, index=df_test.index)
        X_future_scaled = pd.concat([df_test[[self.time_col]], X_future_scaled], axis=1)

        y_true_scaled   = self.scaler_Y.transform(df_test[self.y_cols])
        return df_train_scaled, X_future_scaled, y_true_scaled

    def forecast_using_timegpt(self, df_train: pd.DataFrame, df_test: pd.DataFrame, horizon_len: int, use_exogenous_cols: bool = False):
        """Run TimeGPT forecast for all targets. Returns dict[target] -> forecast series."""
        df_train_scaled, X_future_scaled, y_true_scaled = self._prepare_data(df_train, df_test, horizon_len)
        self.forecast_dict, self.forecast_dfs = {}, {}

        for i, target in enumerate(self.y_cols):
            df_train_target= df_train_scaled[[self.time_col] + self.X_cols + [target]]
            X_df           = X_future_scaled if use_exogenous_cols else None

            forecast_df    = self.client.forecast(df         = df_train_target, # time+ X_past+y
                                                  X_df       = X_df,            # [optional] time + X_future (no y) -> exogenous feature
                                                  h          = horizon_len,     # prediction horizon
                                                  time_col   = self.time_col,
                                                  target_col = target,
                                                  level      = [80, 90],        # confidence interval
                                                  model      = self.model)
            self.forecast_dict[target]= forecast_df["TimeGPT"]
            self.forecast_dfs[target] = forecast_df
            mae = mean_absolute_error(y_true_scaled[:, i], forecast_df["TimeGPT"])
            print(f"MAE for col '{target}': {mae:.4f}")
        return self.forecast_dict, y_true_scaled
