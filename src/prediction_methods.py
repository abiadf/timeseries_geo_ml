from typing import Tuple

import numpy as np
import polars as pl
import pandas as pd
import catboost as cb
import xgboost as xgb

import itertools
from itertools import product
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline

class MultiOutputModelPredictor:
    def __init__(self, device):
        self.device = device
        self.device_str = 'GPU' if self.device.type == 'cuda' else 'CPU'

    def predict_linear_reg(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        model         = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
        y_pred_linreg = model.predict(X_val)
        rmse_linreg   = mean_squared_error(y_val, y_pred_linreg) ** 0.5
        return rmse_linreg, y_pred_linreg

    def predict_linear_reg_ridge(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        model         = MultiOutputRegressor(Ridge(alpha=1.0)).fit(X_train, y_train)
        y_pred_ridge  = model.predict(X_val)
        rmse_ridge    = mean_squared_error(y_val, y_pred_ridge) ** 0.5
        return rmse_ridge, y_pred_ridge


    def predict_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        n_targets = y_train.shape[1]
        y_pred    = np.zeros(y_val.shape)
        
        for i in range(n_targets):
            model = LGBMRegressor(objective       = 'regression',
                                  verbosity       = -1,
                                  n_estimators    = 500, # bigger = better = slower
                                  learning_rate   = 0.15,
                                  max_depth       = 5,  # bigger = better = slower
                                  num_leaves      = 32, # ≈ 2^max_depth
                                  min_data_in_leaf= 15, # lower = more accurate = slower
                                  device          = self.device_str,
                                  feature_fraction=0.7,
                                  bagging_fraction=0.7,
                                  bagging_freq    = 1,)
            model.fit(X_train, y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
            y_pred[:, i] = model.predict(X_val)
        
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        return rmse, y_pred


    # [to remove] seems i duplicated this one below
    def tune_lightgbm_hyperparams_manual(self, X_train, y_train):
        n_targets = y_train.shape[1]
        param_grid = {
            'n_estimators':     [100, 200],
            'learning_rate':    [0.01, 0.05],
            'num_leaves':       [31, 50],
            'max_depth':        [-1, 10],
            'min_data_in_leaf': [20, 50]}
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        combos = list(product(*param_grid.values()))
        results = []

        print(f"Total combos: {len(combos)}")
        for idx, combo in enumerate(combos, 1):
            params = dict(zip(param_grid.keys(), combo))
            print(f"Trying combo {idx}/{len(combos)}: {params}")
            
            val_scores = []
            for train_idx, val_idx in kf.split(X_train):
                # Use .iloc only if X_train is a DataFrame
                if hasattr(X_train, 'iloc'):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                else:
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]

                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                y_pred = np.zeros(y_val.shape)
                for i in range(n_targets):
                    model = LGBMRegressor(objective='regression',
                                        verbosity=-1,
                                        device=self.device_str,
                                        **params)
                    model.fit(X_tr, y_tr[:, i], eval_set=[(X_val, y_val[:, i])],
                            callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
                    y_pred[:, i] = model.predict(X_val)

                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                val_scores.append(rmse)

            avg_rmse = np.mean(val_scores)
            print(f"Avg RMSE: {avg_rmse}")
            results.append((params, avg_rmse))

        best_params = min(results, key=lambda x: x[1])[0]
        print(f"Best params: {best_params}")
        return best_params


    def predict_catboost(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """CatBoost only accepts uppercase 'task_type', beware of that"""
        single_model = cb.CatBoostRegressor(iterations         = 50,
                                            learning_rate      = 0.4,
                                            depth              = 8,
                                            l2_leaf_reg        = 3,
                                            border_count       = 128,
                                            bagging_temperature= 0,
                                            task_type          = 'CPU',
                                            verbose            = 0,
                                            random_seed        = 42)
        multi_model  = MultiOutputRegressor(single_model)
        multi_model.fit(X_train, y_train)

        importances  = np.array([est.get_feature_importance() for est in multi_model.estimators_])
        y_pred_cat   = multi_model.predict(X_val)
        rmse_cat     = mean_squared_error(y_val, y_pred_cat) ** 0.5
        return rmse_cat, y_pred_cat, importances




    def tune_catboost_hyperparams(self, X_train: np.ndarray, y_train: np.ndarray):
        param_grid = {
            'iterations':    [50],
            'learning_rate': [0.38, 0.39, 0.41, 0.42, 0.43],
            'depth':         [8], # higher is better, but takes longer
            'l2_leaf_reg':   [3],
            'border_count':  [64], # higher is better, but takes longer
            'bagging_temperature': [1]}

        best_score  = float('inf')
        best_params = None
        best_model  = None

        combos = list(itertools.product(
            param_grid['iterations'],
            param_grid['learning_rate'],
            param_grid['depth'],
            param_grid['l2_leaf_reg'],
            param_grid['border_count'],
            param_grid['bagging_temperature'],))

        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        total = len(combos)
        for idx, (iterations, learning_rate, depth, l2_leaf_reg, border_count, bagging_temperature) in enumerate(combos, 1):
            print(f"Combo {idx}/{total}: iter={iterations}, lr={learning_rate}, depth={depth}, l2={l2_leaf_reg}, border={border_count}, bagging_temp={bagging_temperature}")
            
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                base_model = cb.CatBoostRegressor(
                    verbose=0,
                    task_type='CPU',
                    thread_count=1,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=depth,
                    l2_leaf_reg=l2_leaf_reg,
                    border_count=border_count,
                    bagging_temperature=bagging_temperature,
                    random_seed=42,
                    early_stopping_rounds=30)
                multi_model = MultiOutputRegressor(base_model)
                multi_model.fit(X_tr, y_tr)
                preds = multi_model.predict(X_val)
                rmse = root_mean_squared_error(y_val, preds)
                cv_scores.append(rmse)

            avg_rmse = np.mean(cv_scores)
            print(f"Avg RMSE: {avg_rmse:.4f}")

            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = {
                    'iterations': iterations,
                    'learning_rate': learning_rate,
                    'depth': depth,
                    'l2_leaf_reg': l2_leaf_reg,
                    'border_count': border_count,
                    'bagging_temperature': bagging_temperature,}
                best_model = multi_model

        print("Best params:", best_params)
        print(f"Best CV RMSE: {best_score:.4f}")
        return best_model

    def tune_lightgbm_hyperparams(self, X_train: np.ndarray, y_train: np.ndarray):
        param_grid = {
            'n_estimators':     [50],
            'learning_rate':    [0.15, 0.2, 0.25, 0.35],#, 0.4],
            'max_depth':        [6],
            'reg_lambda':       [3],
            'num_leaves':       [64],# usually ~2^depth
            'bagging_fraction': [1.0],}   # 1 = no bagging

        best_score  = float('inf')
        best_params = None
        best_model  = None

        combos = list(itertools.product(
            param_grid['n_estimators'],
            param_grid['learning_rate'],
            param_grid['max_depth'],
            param_grid['reg_lambda'],
            param_grid['num_leaves'],
            param_grid['bagging_fraction'],))

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        total = len(combos)

        for idx, (n_estimators, learning_rate, max_depth, reg_lambda, num_leaves, bagging_fraction) in enumerate(combos, 1):
            print(f"Combo {idx}/{total}: est={n_estimators}, lr={learning_rate}, depth={max_depth}, lambda={reg_lambda}, leaves={num_leaves}, bag_frac={bagging_fraction}")
            
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                base_model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    reg_lambda=reg_lambda,
                    num_leaves=num_leaves,
                    bagging_fraction=bagging_fraction,
                    subsample_freq=1,
                    verbose=-1,
                    n_jobs=1,
                    random_state=42)

                multi_model = MultiOutputRegressor(base_model)
                multi_model.fit(X_tr, y_tr)
                preds = multi_model.predict(X_val)
                rmse = root_mean_squared_error(y_val, preds)
                cv_scores.append(rmse)

            avg_rmse = np.mean(cv_scores)
            print(f"Avg RMSE: {avg_rmse:.4f}")

            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = {
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'reg_lambda': reg_lambda,
                    'num_leaves': num_leaves,
                    'bagging_fraction': bagging_fraction}
                best_model = multi_model

        print("Best params:", best_params)
        print(f"Best CV RMSE: {best_score:.4f}")
        return best_model


    @staticmethod
    def predict_xgboost(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', verbosity=0))
        model.fit(X_train, y_train)
        y_pred_xgb = model.predict(X_val)
        rmse_xgb = mean_squared_error(y_val, y_pred_xgb) ** 0.5
        return rmse_xgb, y_pred_xgb

    @staticmethod
    def predict_randomforest(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        rf = RandomForestRegressor(n_estimators = 100,
                                   max_depth    = 15,
                                   max_features = 'sqrt',
                                   min_samples_leaf=10,
                                   n_jobs       = -1,
                                   random_state = 42)
        rf_model  = MultiOutputRegressor(rf)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_val)
        rmse_rf   = mean_squared_error(y_val, y_pred_rf) ** 0.5
        return rmse_rf, y_pred_rf

    @staticmethod
    def predict_hgb(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        model = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100))
        model.fit(X_train, y_train)
        y_pred_hgb = model.predict(X_val)
        rmse_hgb = mean_squared_error(y_val, y_pred_hgb) ** 0.5
        return rmse_hgb, y_pred_hgb

    @staticmethod
    def predict_elasticnet(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, np.ndarray]:
        imputer = SimpleImputer(strategy="mean")
        base_model = make_pipeline(imputer, ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000))
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        y_pred_elas = model.predict(X_val)
        rmse_elas = mean_squared_error(y_val, y_pred_elas) ** 0.5
        return rmse_elas, y_pred_elas


class DataPreprocessor:
    def __init__(self):
        self.y_scaler = StandardScaler() #RobustScaler()
        self.x_scaler = StandardScaler() #RobustScaler()

    def join_logs_and_wafer_df(self, log_df: pl.DataFrame, wafer_df: pl.DataFrame, y_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        X_full = log_df.join(wafer_df, on="marathon_run", how="inner", suffix="_df2")
        X_full_pd = X_full.to_pandas()
        y_full_pd = y_df.sort("marathon_run").drop("marathon_run").to_pandas()
        return X_full_pd, y_full_pd

    # def _drop_non_numeric_cols_from_df(self, df):
    #     if type(df) == pl.DataFrame:
    #         return df.select(pl.all().filter(lambda s: s.dtype.is_numeric()))
    #     elif type(df) == pd.DataFrame:
    #         return df.select_dtypes(include=[np.number])

    def drop_certain_cols_from_df(self, df, cols_to_drop):
        """Drop a set of cols from df. Use this function when unsure of the df type"""
        if isinstance(df, pl.DataFrame):
            return df.drop([col for col in cols_to_drop if col in df.columns])
        elif isinstance(df, pd.DataFrame):
            existing_cols = [col for col in cols_to_drop if col in df.columns]
            return df.drop(columns=existing_cols)
        else:
            raise TypeError("Unsupported DataFrame type")

    def scale_and_split_data(self, X_full_pd: pd.DataFrame, y_full_pd: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, StandardScaler]:
        test_size = 0.2
        y_full_scaled_np                     = self.y_scaler.fit_transform(y_full_pd)
        X_train, X_val, y_train_np, y_val_np = train_test_split(X_full_pd, y_full_scaled_np, test_size=test_size, random_state=42)

        cols_to_scale = [c for c in X_train.select_dtypes(include=np.number).columns]
        # print("Cols to scale:", cols_to_scale)

        X_train.loc[:, cols_to_scale] = self.x_scaler.fit_transform(X_train[cols_to_scale])
        X_val.loc[:, cols_to_scale]   = self.x_scaler.transform(X_val[cols_to_scale])

        return X_train, y_train_np, X_val, y_val_np, self.y_scaler


    def scale_per_wafer_and_split_data(self, X_full_pd: pd.DataFrame, y_full_pd: pd.DataFrame, wafer_col="wafer", test_size=0.2):
        """Scale features and targets per wafer to avoid data leakage across wafers.
        Splits data into train/validation sets first, then fits scalers only on train wafers and applies them to validation wafers.

        WARNING: This function's output is NOT compatible with sklearn's cross_val_score,
        because cross_val_score does not refit scalers per fold, leading to data leakage if used with this per-wafer scaling.

        Use custom group-aware CV with per-fold scaling to avoid leakage in cross-validation"""

        # Split first (to avoid info leak)
        X_train, X_val, y_train, y_val = train_test_split(X_full_pd, y_full_pd, test_size=test_size, random_state=42)

        # Numeric columns except wafer_col for X
        cols_to_scale_X = [c for c in X_train.select_dtypes(include="number").columns if c != wafer_col]
        # Numeric columns for y
        cols_to_scale_y = [c for c in y_train.select_dtypes(include="number").columns if c != wafer_col]

        wafer_x_scalers = {}
        wafer_y_scalers = {}

        def scale_df_group(df, cols, scaler=None):
            if scaler is None:
                scaler = StandardScaler()
                scaled_vals = scaler.fit_transform(df[cols])
            else:
                scaled_vals = scaler.transform(df[cols])
            df.loc[:, cols] = scaled_vals
            return df, scaler

        # Scale X_train and y_train per wafer
        X_train_scaled_list = []
        y_train_scaled_list = []

        for wafer_id, X_grp in X_train.groupby(wafer_col):
            y_grp = y_train.loc[X_grp.index]

            X_grp_scaled, x_scaler = scale_df_group(X_grp.copy(), cols_to_scale_X)
            y_grp_scaled, y_scaler = scale_df_group(y_grp.copy(), cols_to_scale_y)

            wafer_x_scalers[wafer_id] = x_scaler
            wafer_y_scalers[wafer_id] = y_scaler

            X_train_scaled_list.append(X_grp_scaled)
            y_train_scaled_list.append(y_grp_scaled)

        X_train_scaled = pd.concat(X_train_scaled_list).sort_index()
        y_train_scaled = pd.concat(y_train_scaled_list).sort_index()

        # Scale X_val and y_val per wafer using stored scalers
        X_val_scaled_list = []
        y_val_scaled_list = []

        for wafer_id, X_grp in X_val.groupby(wafer_col):
            y_grp = y_val.loc[X_grp.index]

            x_scaler = wafer_x_scalers.get(wafer_id)
            y_scaler = wafer_y_scalers.get(wafer_id)

            if x_scaler is not None:
                X_grp_scaled, _ = scale_df_group(X_grp.copy(), cols_to_scale_X, scaler=x_scaler)
            else:
                X_grp_scaled = X_grp.copy()

            if y_scaler is not None:
                y_grp_scaled, _ = scale_df_group(y_grp.copy(), cols_to_scale_y, scaler=y_scaler)
            else:
                y_grp_scaled = y_grp.copy()

            X_val_scaled_list.append(X_grp_scaled)
            y_val_scaled_list.append(y_grp_scaled)

        X_val_scaled = pd.concat(X_val_scaled_list).sort_index()
        y_val_scaled = pd.concat(y_val_scaled_list).sort_index()

        # Return X,y as DataFrames, but y can be converted to np.ndarray if needed
        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, wafer_x_scalers, wafer_y_scalers
