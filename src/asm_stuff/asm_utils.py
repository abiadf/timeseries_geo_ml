import os
import time
from typing import Union, List
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from skdim.id import MLE
from typing import Tuple
from category_encoders import TargetEncoder
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

class Basics:

    @staticmethod
    def count_missing_values_in_df(df) -> None:
        """Count NaNs (floats) and nulls (all types) in pandas or Polars df"""

        if isinstance(df, pd.DataFrame):
            float_cols = [col for col, dt in df.dtypes.items() if pd.api.types.is_float_dtype(dt)]

            # NaNs in float columns
            nan_per_col    = df[float_cols].isna().sum()
            cols_with_nan  = (nan_per_col > 0).sum()
            rows_with_nan  = df[float_cols].isna().any(axis=1).sum()
            total_nan_cells= nan_per_col.sum()

            # Nulls in all columns (in pandas, NaN == null)
            null_per_col    = df.isnull().sum()
            cols_with_nulls = (null_per_col > 0).sum()
            rows_with_nulls = df.isnull().any(axis=1).sum()
            total_null_cells= null_per_col.sum()
        elif isinstance(df, pl.DataFrame):
            float_cols = [col for col, dt in zip(df.columns, df.dtypes) if dt.is_float()]
            if float_cols:
                nan_per_col   = df.select([pl.col(col).is_nan().sum().alias(col) for col in float_cols])
                cols_with_nan = sum(val > 0 for val in nan_per_col.row(0))
                rows_with_nan = df.filter(
                    pl.fold(False, lambda acc, e: acc | e, [pl.col(c).is_nan() for c in float_cols])).height
                total_nan_cells= nan_per_col.to_series().sum()
            else:
                cols_with_nan = rows_with_nan = total_nan_cells = 0
            null_per_col    = df.select([pl.col(col).is_null().sum().alias(col) for col in df.columns])
            cols_with_nulls = sum(val > 0 for val in null_per_col.row(0))
            rows_with_nulls = df.filter(
                pl.fold(False, lambda acc, e: acc | e, [pl.col(c).is_null() for c in df.columns])).height
            total_null_cells= null_per_col.to_series().sum()
        else:
            raise TypeError("Unsupported DataFrame type. Pass pandas or Polars DataFrame.")
        print('---------')
        print(f"NaNs: in {cols_with_nan} cols, {rows_with_nan} rows, {total_nan_cells} cells in total")
        print(f"Nulls: in {cols_with_nulls} cols, {rows_with_nulls} rows, {total_null_cells} cells in total")

    @staticmethod
    def remove_constant_valued_cols(df):
        """Drop numeric columns with a single unique value (constant-valued columns)."""
        if isinstance(df, pl.DataFrame):
            numeric_cols  = df.select(pl.selectors.numeric()).columns
            constant_cols = [col for col in numeric_cols if df[col].n_unique() == 1]
            return df.drop(constant_cols)
        elif isinstance(df, pd.DataFrame):
            constant_cols = [col for col in df.select_dtypes(include='number').columns
                            if df[col].nunique() == 1]
            return df.drop(columns=constant_cols)
        else:
            raise TypeError("Unsupported DataFrame type")

    @staticmethod
    def drop_shared_high_nan_cols(df1: pd.DataFrame, df2: pd.DataFrame, threshold: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drop columns from both df1 and df2 if more than `threshold` fraction of values are NaN in either."""
        nan_frac_1   = df1.isna().mean()
        nan_frac_2   = df2.isna().mean()
        cols_to_drop = nan_frac_1[(nan_frac_1 > threshold)].index.union(nan_frac_2[(nan_frac_2 > threshold)].index)
        return df1.drop(columns=cols_to_drop), df2.drop(columns=cols_to_drop)

    @staticmethod
    def drop_and_impute_nan_cols(df, threshold=0.5, impute: str | None = None):
        """Drop cols with NaN ratio > threshold, then optionally fill remaining NaNs with col's median/mean/0 (see last line in function)
        - 'mean': fill with column mean
        - 'median': fill with column median
        - 'zero': fill with 0
        - None: leave NaNs as-is
        NOTE: catboost handles NaNs (so leave them, otherwise we lose info), but methods like PCA cannot"""
        nan_ratio_df = df.isna().mean()
        cols_to_keep = nan_ratio_df[nan_ratio_df <= threshold].index

        df = df[cols_to_keep]
        if impute == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        elif impute == 'median':
            return df.fillna(df.median(numeric_only=True))
        elif impute == 'zero':
            return df.fillna(0)
        elif impute is None:
            return df
        else:
            raise ValueError(f"Unknown imputation method: {impute}")

    @staticmethod
    def plot_all_columns_in_df(df):
        """Given a df, plot all its cols into a figure"""

        num_cols  = df.shape[1]
        num_rows  = (num_cols // 10) + 1  # 10 plots per. figure row
        fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
        axes      = axes.flatten()

        for i, col in enumerate(df.columns):
            axes[i].plot(df[col])
            axes[i].set_title(str(col), fontsize=6)
            axes[i].tick_params(labelsize=4)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def estimate_dataset_dimensionality(dataset, n_neighbors=10):
        """Estimate intrinsic dimensionality using scikit-dimension (ie best latent size)
        input: dataset (pd or pl df, or np array), and n_neighbors (larger for bigger dataset)
        output: estimated dataset dimensionality"""
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.select_dtypes(include=[np.number])
            X = dataset.to_numpy()
        elif isinstance(dataset, pl.DataFrame):
            dataset = dataset.select(pl.col(pl.NUMERIC_DTYPES))
            X = dataset.to_numpy()
        elif isinstance(dataset, (np.ndarray,)):
            X = dataset
        else:
            raise TypeError("Unsupported dataset type")
        return MLE().fit(X, n_neighbors=n_neighbors).dimension_

    @staticmethod
    def apply_target_encoding_to_df(X: pd.DataFrame, y: pd.DataFrame, col_to_encode: str, target_name: str) -> Tuple[pd.DataFrame, TargetEncoder]:
        """Apply target encoding to a categorical column using the mean of y
        If y is multi-output, the row-wise mean is used as the target
        Returns the updated df with the new encoded column + the fitted encoder
        NOTE: make sure to fit target encoding to trainset, then apply to test set (prevents leakage)"""
        target         = y.mean(axis=1) if y.ndim > 1 else y
        te_model       = TargetEncoder(cols = [col_to_encode])
        X[target_name] = te_model.fit_transform(X[col_to_encode], target)
        return X.drop(columns = [col_to_encode]), te_model

    # to remove
    @staticmethod
    def save_catboost_models(catboost_models, model_dir) -> None:
        """Saves each CatBoost model in MultiOutputRegressor to a .cbm file. We add a target index to the filename
        so we can load them in the right order later
        - catboost_models: MultiOutputRegressor containing CatBoostRegressor models"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        for i, model in enumerate(catboost_models.estimators_):
            model: CatBoostRegressor
            model.save_model(f"{model_dir}/catboost_target_{i}.cbm")

    @staticmethod
    def save_catboost_models(catboost_models: Union[List[CatBoostRegressor], object], model_dir: str) -> None:
        """Save CatBoost models to disk. Supports either:
            - A list of CatBoostRegressor models (one per target), or
            - A MultiOutputRegressor-like object with an 'estimators_' attribute
        Each model is saved as 'catboost_target_i.cbm' in `model_dir`.
        - catboost_models: List of CatBoostRegressor or MultiOutputRegressor-like object.
        - model_dir: Directory path where models will be saved."""

        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        if isinstance(catboost_models, list):
            models = catboost_models
        elif hasattr(catboost_models, "estimators_"):
            models = catboost_models.estimators_
        else:
            raise TypeError("Unsupported model type")

        for i, model in enumerate(models):
            model.save_model(f"{model_dir}/catboost_target_{i}.cbm")

    @staticmethod
    def load_all_catboost_models_in_dir(model_dir) -> list[CatBoostRegressor]:
        """Loads Catboost models from .cbm files in model_dir and returns a list of models
        This function takes ALL .cbm files in the dir (easier than specifying n_targets). We save and
        load by name-sorted to preserve model order (ie, model for y0 is #1 to be loaded, y10 is #11...)
        - model_dir: directory containing the .cbm files"""
        models_list = []
        files       = sorted(f for f in os.listdir(model_dir) if f.endswith(".cbm"))
        for file in files:
            model = CatBoostRegressor()
            model.load_model(os.path.join(model_dir, file))
            models_list.append(model)
        return models_list

    @staticmethod
    def optimize_df_memory(df: pd.DataFrame, nan_threshold: float = 0.9) -> pd.DataFrame:
        """Downcast numeric types, convert low-cardinality objects to category, 
        and convert sparse-like columns to SparseDtype to reduce memory"""

        df = df.copy()
        for col, col_data in df.items():
            if pd.api.types.is_integer_dtype(col_data):
                df[col] = pd.to_numeric(col_data, downcast='integer')
            elif pd.api.types.is_float_dtype(col_data):
                df[col] = pd.to_numeric(col_data, downcast='float')

            # Convert object -> category if few unique values
            elif col_data.dtype == 'object':
                num_unique = col_data.nunique(dropna=False)
                num_total  = len(col_data)
                if num_unique / num_total < 0.5:
                    df[col] = col_data.astype('category')

            # Convert sparse-like (mostly NaN or 0) to sparse
            if df[col].isna().sum() / len(df[col]) > nan_threshold or (df[col] == 0).sum() / len(df[col]) > nan_threshold:
                df[col] = pd.arrays.SparseArray(df[col])
        return df
