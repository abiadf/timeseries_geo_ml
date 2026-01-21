"""Contains the logic for processing a dataset to X and y"""
"🇮🇳 india ocean wave timeseries (https://www.kaggle.com/competitions/ml-for-oceanography/data)"
"🇭🇺 Szeged (https://www.kaggle.com/datasets/budincsevity/szeged-weather)"
"longterm weather (https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting)"
"💨 wind power (https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)"
"🚁 wind turbine power (https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting?select=Location1.csv)"
"⚡️ Electric power data (https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)"

import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Dict, List, Literal, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass

import numexpr as ne # makes numpy operations faster
import numpy as np
import pandas as pd
import polars as pl

from ucimlrepo import fetch_ucirepo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    # print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")


def process_dataset_given_filename(file_loc, y_cols: List[str]):
    """GIven a dataset file location and target column names, process and return X and y"""
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    X      = pd.read_csv(file_loc)
    y      = X[y_cols].values
    X      = X.drop(columns=y_cols)
    return X, y

def process_china_weather_dataset(file_location: str, y_col_indices: list[int]):
    """Process China weather tensor dataset."""
    if not all(isinstance(i, int) for i in y_col_indices):
        raise TypeError("y_col_indices must be integer feature indices")
    page_choice = 7
    china_weather_array = np.load(file_location).transpose(0, 2, 1)
    mask = np.ones(china_weather_array.shape[2], dtype=bool)
    mask[y_col_indices] = False
    X_cut = china_weather_array[:, :, mask]
    y_cut = china_weather_array[:, :, y_col_indices]
    X = pd.DataFrame(X_cut[page_choice])
    y = y_cut[page_choice]
    return X, y

def get_household_power_consumption(file_loc: str, y_cols: list[str]):
    """Load household power consumption dataset."""
    target = y_cols[0]
    if os.path.exists(file_loc):
        df = pl.read_parquet(file_loc)
    else:
        ds = fetch_ucirepo(id=235)
        df_pd = ds.data.original
        cols_to_fix = [c for c in df_pd.columns if c not in ["Date", "Time"]]
        for col in cols_to_fix:
            df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')
        df = pl.from_pandas(df_pd)
        df.write_parquet(file_loc)

    df = df.filter(pl.col(target).is_not_null())
    y  = df.get_column(target)
    X  = df.drop(target).fill_null(0)
    return X.to_pandas(), y.to_pandas()


dataset_dict = {"szeged_weather":    {"file_loc": "../public_datasets/2D/tabular/szeged_weather.csv",
                                      "y_cols": ["Temperature (C)"],
                                      "function": process_dataset_given_filename},
                "india_ocean_waves": {"file_loc": "../public_datasets/2D/tabular/india ocean train.csv",
                                      "y_cols": ["WVHT(m)"],
                                      "function": process_dataset_given_filename},
                "longterm_weather":  {"file_loc": '../public_datasets/2D/tabular/longterm_weather/longterm_weather.csv',
                                      "y_cols": ["rain"],
                                      "function": process_dataset_given_filename},
                "panama":            {"file_loc": f"../public_datasets/3D/panama/train.csv",
                                      "y_cols": ["T2M_toc"],
                                      "function": process_dataset_given_filename},
                "wind_power":        {"file_loc": f"../public_datasets/2D/tabular/wind_power/wind_power_dataset.csv",
                                      "y_cols": ["LV ActivePower (kW)"],
                                      "function": process_dataset_given_filename},
                "turbine_power":     {"file_loc": "../public_datasets/2D/tabular/turbine_power/Location1.csv",
                                      "y_cols": ["Power"],
                                      "function": process_dataset_given_filename},
                "electric_power":    {"file_loc": "../public_datasets/2D/tabular/electric_power/household_power.parquet",
                                      "y_cols": ["Global_active_power"],
                                      "function": get_household_power_consumption},
                "china_weather":    {"file_loc": "../public_datasets/3D/china_weather/china_weather_000.npy",
                                      "y_cols": [7, 10], #[4, 5, 6, 7, 10]
                                      "function": process_china_weather_dataset}
                }

