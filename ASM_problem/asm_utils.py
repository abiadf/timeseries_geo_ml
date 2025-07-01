import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from skdim.id import MLE

def count_missing_values_in_df(df) -> None:
    """Count NaNs (floats) and nulls (all types) in pandas or Polars DataFrame."""

    if isinstance(df, pd.DataFrame):
        float_cols = [col for col, dt in df.dtypes.items() if pd.api.types.is_float_dtype(dt)]

        # NaNs in float columns
        nan_per_col   = df[float_cols].isna().sum()
        cols_with_nan = (nan_per_col > 0).sum()
        rows_with_nan = df[float_cols].isna().any(axis=1).sum()
        total_nans    = nan_per_col.sum()

        # Nulls in all columns (in pandas, NaN == null)
        null_per_col    = df.isnull().sum()
        cols_with_nulls = (null_per_col > 0).sum()
        rows_with_nulls = df.isnull().any(axis=1).sum()
        total_nulls     = null_per_col.sum()

    elif isinstance(df, pl.DataFrame):
        float_cols = [col for col, dt in zip(df.columns, df.dtypes) if dt.is_float()]

        if float_cols:
            nan_per_col   = df.select([pl.col(col).is_nan().sum().alias(col) for col in float_cols])
            cols_with_nan = sum(val > 0 for val in nan_per_col.row(0))
            rows_with_nan = df.filter(
                pl.fold(False, lambda acc, e: acc | e, [pl.col(c).is_nan() for c in float_cols])).height
            total_nans    = nan_per_col.to_series().sum()
        else:
            cols_with_nan = rows_with_nan = total_nans = 0

        null_per_col    = df.select([pl.col(col).is_null().sum().alias(col) for col in df.columns])
        cols_with_nulls = sum(val > 0 for val in null_per_col.row(0))
        rows_with_nulls = df.filter(
            pl.fold(False, lambda acc, e: acc | e, [pl.col(c).is_null() for c in df.columns])).height
        total_nulls     = null_per_col.to_series().sum()

    else:
        raise TypeError("Unsupported DataFrame type. Pass pandas or Polars DataFrame.")
    print('---------')
    print(f"NaNs: in {cols_with_nan} cols, {rows_with_nan} rows, {total_nans} in total")
    print(f"Nulls: in {cols_with_nulls} cols, {rows_with_nulls} rows, {total_nulls} in total")

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