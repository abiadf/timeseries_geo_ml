"""Contains the logic for processing a dataset to X and y"""
"🇮🇳 india ocean wave timeseries (https://www.kaggle.com/competitions/ml-for-oceanography/data)"
"🇭🇺 Szeged (https://www.kaggle.com/datasets/budincsevity/szeged-weather)"
"🌦️ longterm weather (https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting)"
"💨 wind power (https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)"
"🚁 turbine power (https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting?select=Location1.csv)"
"⚡️ Electric power data (https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)"

import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import List
import logging
from dataclasses import dataclass
import requests, re, io

import numexpr as ne # makes numpy operations faster
import numpy as np
import pandas as pd
import polars as pl

from ucimlrepo import fetch_ucirepo

import torch
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
    page_choice         = 7
    china_weather_array = np.load(file_location).transpose(0, 2, 1)
    mask                = np.ones(china_weather_array.shape[2], dtype=bool)
    mask[y_col_indices] = False
    X_cut = china_weather_array[:, :, mask]
    y_cut = china_weather_array[:, :, y_col_indices]
    X     = pd.DataFrame(X_cut[page_choice])
    y     = y_cut[page_choice]
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

def clean_beijing_data(X_orig: pd.DataFrame, y_orig: np.ndarray):
    wd_to_deg = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}

    X_orig["wd_deg"] = X_orig["wd"].map(wd_to_deg)
    X_orig.drop(columns=["wd"], inplace=True)

    X_orig.infer_objects(copy=False)
    for col in X_orig.select_dtypes(include="object").columns:
        X_orig[col] = pd.to_numeric(X_orig[col], errors="coerce")

    X_orig = X_orig.interpolate(method="linear", limit_direction="both")
    y_orig = np.interp(np.arange(len(y_orig)), np.where(~np.isnan(y_orig))[0], y_orig[~np.isnan(y_orig)])
    print(f"🧹 Cleaned Beijing data")
    return X_orig, y_orig


class NasaData:
    """Outputs a DataFrame with lunar ephemeris data from NASA JPL Horizons system.
        # Target y: delta (= distance between the Earth center and the Moon center.)
        # Units: Astronomical Units (AU)
        # Description: Geometric distance between Earth and Moon centers.
        # Scale: Fluctuates ~0.0024 to 0.0027 AU per lunar cycle.

    LUNAR DATASET FEATURES & UNITS:
    - date: Universal Time (UTC)
    - R.A._(ICRF): Right Ascension [Decimal Degrees 0-360]
    - DEC__(ICRF): Declination [Decimal Degrees -90 to +90]
    - delta: Earth-Moon Distance (Target y) [Astronomical Units (AU)]
    - S-T-O: Sun-Target-Observer (Phase) Angle [Decimal Degrees 0-180]
    - ObsEcLon/Lat: Ecliptic Longitude/Latitude [Decimal Degrees]
    - GlxLon/Lat: Galactic Longitude/Latitude [Decimal Degrees]"""

    @staticmethod
    def hms_to_deg(hms_str):
        if pd.isna(hms_str): return np.nan
        parts = re.split(r'\s+', str(hms_str).strip())
        if len(parts) != 3: return np.nan
        h, m, s = map(float, parts)
        return (h + m/60 + s/3600) * 15

    @staticmethod
    def dms_to_deg(dms_str):
        if pd.isna(dms_str): return np.nan
        parts = re.split(r'\s+', str(dms_str).strip())
        if len(parts) != 3: return np.nan
        d, m, s = map(float, parts)
        sign = -1 if '-' in parts[0] else 1
        return sign * (abs(d) + m/60 + s/3600)

    @staticmethod
    def encode_angles(df, angle_cols):
        for col in angle_cols:
            rad = np.radians(df[col])
            df[f'{col}_sin'] = np.sin(rad)
            df[f'{col}_cos'] = np.cos(rad)
        return df.drop(columns=angle_cols)

    @staticmethod
    def moon_periodic_dataset(start: str, stop: str, step: str = "1 d") -> pd.DataFrame:
        """Freatures from nasa page (https://ssd.jpl.nasa.gov/horizons/app.html#/), "table settings":
        ['Date__(UT)__HR:MN', 'R.A._(ICRF)', 'DEC__(ICRF)', 'R.A.__(a-app)', 'DEC_(a-app)', 'dRA*cosD', 'd(DEC)/dt', 'Azi_(a-app)',
        'Elev_(a-app)', 'Illu%', 'hEcl-Lon', 'hEcl-Lat', 'delta', 'deldot', 'S-O-T', '/r', 'S-T-O', 'PsAng', 'PsAMV', 'ObsEcLon',
        'ObsEcLat', 'GlxLon', 'GlxLat', 'datetime'] """

        url = "https://ssd.jpl.nasa.gov/api/horizons.api"
        # 1: RA/DEC, 20: Range, 24: S-T-O, 31: Ecliptic, 33: Galactic
        q_list = "1,20,24,31,33"    
        params = {
            "format": "json", "COMMAND": "'301'", "CENTER": "'500@399'",
            "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER", 
            "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'",
            "STEP_SIZE": f"'{step}'", "QUANTITIES": f"'{q_list}'",
            "CSV_FORMAT": "YES", "OBJ_DATA": "NO"}

        r = requests.get(url, params=params)
        txt = r.json().get("result", "")
        data_match = re.search(r"\$\$SOE(.*)\$\$EOE", txt, flags=re.S)
        if not data_match: return pd.DataFrame()
        
        header_section = txt.split("$$SOE")[0].strip()
        header_line = [l for l in header_section.splitlines() if "," in l][-1]
        raw_cols = [c.strip() for c in header_line.split(",")]

        clean_cols = []
        for i, name in enumerate(raw_cols):
            n = name.strip() if name.strip() else f"empty_{i}"
            if n in clean_cols: n = f"{n}_{i}"
            clean_cols.append(n)

        df = pd.read_csv(io.StringIO(data_match.group(1).strip()), names=clean_cols, index_col=False)
        
        for col in df.columns:
            if 'R.A.' in col: df[col] = df[col].apply(NasaData.hms_to_deg)
            elif 'DEC' in col: df[col] = df[col].apply(NasaData.dms_to_deg)

        # Cleanup specific columns requested and NASA artifacts
        exclude = ['empty', '/*', '/r', 'PsAng', 'PsAMV', 'Azi', 'Elev', 
                   'deldot', 'a-app', 'Illu%', 'S-O-T']
        df = df.drop(columns=[c for c in df.columns if any(x in c for x in exclude)]).dropna(axis=1, how='all')
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        return df

# how to run
# df = NasaData.moon_periodic_dataset("2020-01-01", "2026-01-01", "12 h")
# cols_to_fix = ['R.A._(ICRF)', 'ObsEcLon', 'GlxLon'] #have this 0-360 deg range
# df = NasaData.encode_angles(df, cols_to_fix)
# df.to_csv("../public_datasets/2D/tabular/moon_data_2020_2026.csv", index=False)




dataset_dict = {"szeged_weather":    {"file_loc": "../public_datasets/2D/tabular/szeged_weather.csv",
                                      "y_cols": ["Temperature (C)"],
                                      "function": process_dataset_given_filename},
                "india_ocean_waves": {"file_loc": "../public_datasets/2D/tabular/india ocean train.csv",
                                      "y_cols": ["WVHT(m)"],
                                      "function": process_dataset_given_filename},
                "longterm_weather":  {"file_loc": '../public_datasets/2D/tabular/longterm_weather/longterm_weather.csv',
                                      "y_cols": ["VPact"],
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
                "china_weather":     {"file_loc": "../public_datasets/3D/china_weather/china_weather_000.npy",
                                      "y_cols": [7, 10], #[4, 5, 6, 7, 10]
                                      "function": process_china_weather_dataset},
                "gas":               {"file_loc": f"../public_datasets/3D/gas_emissions/gt_2011.csv",
                                      "y_cols": ["CO","NOX"],
                                      "function": process_dataset_given_filename},
                "beijing":           {"file_loc": f"../public_datasets/3D/beijing/PRSA_Data_Changping_20130301-20170228.csv",
                                      "y_cols": ["PM2.5"],
                                      "function": process_dataset_given_filename,
                                      "processing": clean_beijing_data},
                "nasa_moon":         {"file_loc": f"../public_datasets/2D/tabular/moon_data_2020_2026.csv",
                                      "y_cols": ["delta"],
                                      "function": process_dataset_given_filename},
                }


