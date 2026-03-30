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
from statsmodels.tsa.statespace.sarimax import SARIMAX

class BaseForecaster:
    """Abstract forecaster with preprocessing and scaling."""
    def __init__(self, df: pd.DataFrame, time_col: str, y_cols: list):
        self.df       = df.copy()
        self.time_col = time_col
        self.y_cols   = y_cols
        
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        self.X_cols        = None
        self.forecast_dict = {}
        self.forecast_dfs  = {}

    def _prepare_data(self, df_train: pd.DataFrame, df_test: pd.DataFrame, horizon_len: int):
        """Scale features and split into train/test sets."""
        if self.X_cols is None:
            self.X_cols = [c for c in df_train.columns if c not in [self.time_col] + self.y_cols]

        df_train_scaled              = df_train.copy()
        df_train_scaled[self.X_cols] = self.scaler_X.fit_transform(df_train[self.X_cols])
        df_train_scaled[self.y_cols] = self.scaler_Y.fit_transform(df_train[self.y_cols])
        
        df_test         = df_test.iloc[:horizon_len].copy()
        X_future_scaled = pd.DataFrame(self.scaler_X.transform(df_test[self.X_cols]),
                                       columns=self.X_cols, index=df_test.index)
        X_future_scaled = pd.concat([df_test[[self.time_col]], X_future_scaled], axis=1)
        y_true_scaled   = self.scaler_Y.transform(df_test[self.y_cols])
        return df_train_scaled, X_future_scaled, y_true_scaled

        # Scale test set WITHOUT slicing
        # df_test_scaled = df_test.copy()
        # df_test_scaled[self.X_cols] = self.scaler_X.transform(df_test[self.X_cols])
        # y_true_scaled = self.scaler_Y.transform(df_test[self.y_cols])
        # return df_train_scaled, df_test_scaled, y_true_scaled

    def forecast(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forecasting logic.")


class SARIMAXForecaster(BaseForecaster):
    """SARIMAX-based forecasting, inherited from BaseForecaster"""
    def forecast_sarimax(self, df_train, df_test, horizon_len, order=(1,0,0), seasonal_order=(0,0,0,0),
                         use_exogenous_cols: bool = True):
        df_train_scaled, X_future_scaled, y_true_scaled = self._prepare_data(df_train, df_test, horizon_len)
        self.forecast_dict, self.forecast_dfs = {}, {}

        exog_train  = df_train_scaled[self.X_cols] if (self.X_cols and use_exogenous_cols) else None
        exog_future = X_future_scaled[self.X_cols] if (self.X_cols and use_exogenous_cols) else None

        for i, target in enumerate(self.y_cols):
            model    = SARIMAX(df_train_scaled[target], exog=exog_train, order=order, seasonal_order=seasonal_order)
            results  = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon_len, exog=exog_future).predicted_mean
            self.forecast_dict[target] = forecast.values
            self.forecast_dfs[target]  = pd.DataFrame({self.time_col: df_test[self.time_col].iloc[:horizon_len],
                                                    "SARIMAX": forecast})
            print(f"MAE for '{target}': {mean_absolute_error(y_true_scaled[:, i], forecast):.4f}")
        return self.forecast_dict, y_true_scaled

class TimeGPTForecaster(BaseForecaster):
    """TimeGPT-based forecasting, inherited from BaseForecaster"""
    def __init__(self, df, time_col, y_cols, api_key, model="timegpt-1-long-horizon"):
        super().__init__(df, time_col, y_cols)
        self.model  = model
        self.client = NixtlaClient(api_key=api_key)

    def forecast_timegpt(self, df_train, df_test, horizon_len, use_exogenous_cols=False):
        df_train_scaled, X_future_scaled, y_true_scaled = self._prepare_data(df_train, df_test, horizon_len)
        self.forecast_dict, self.forecast_dfs = {}, {}

        for i, target in enumerate(self.y_cols):
            df_train_target = df_train_scaled[[self.time_col] + self.X_cols + [target]]
            X_df            = X_future_scaled if use_exogenous_cols else None

            forecast_df = self.client.forecast(df=df_train_target, X_df=X_df, h=horizon_len,
                                               time_col=self.time_col, target_col=target,
                                               level=[80, 90], model=self.model)
            self.forecast_dict[target] = forecast_df["TimeGPT"]
            self.forecast_dfs[target]  = forecast_df
            print(f"MAE for '{target}': {mean_absolute_error(y_true_scaled[:, i], forecast_df['TimeGPT']):.4f}")
        return self.forecast_dict, y_true_scaled

