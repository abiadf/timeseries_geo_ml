"""Dictionaties for datasets to use in main code"""

dataset_modality_dict = {"longterm_weather":  "timeseries",
                         "electric_power":    "timeseries",
                         "china_weather":     "timeseries",
                         "panama":            "timeseries",
                         "wind_power":        "timeseries",
                         "india_ocean_waves": "timeseries",
                         "szeged_weather":    "timeseries",
                         "turbine_power":     "timeseries",
                         "imu_gyro":          "timeseries",}

cols_to_drop = {"longterm_weather":  ["date"],
                "electric_power":    ["Date", "Time"],
                "china_weather":     [],
                "cali_housing":      [],
                "bike_sharing":      ["dteday"],
                "forest_fires":      [],
                "panama":            ["datetime"],
                "wind_power":        ["Date/Time"],
                "india_ocean_waves": ['ID', '#YY', 'MM', 'DD', 'hh', 'mm'],
                "szeged_weather":    ["Precip Type","Summary", "Formatted Date", "Apparent Temperature (C)", "Loud Cover", "Daily Summary"],
                "turbine_power":     ["Time"],
                "imu_gyro":          [],}

periodic_threshold_dict = {"longterm_weather":  0.52,
                           "electric_power":    0.5,
                           "china_weather":     0.3,
                           "cali_housing":      0.3,
                           "bike_sharing":      0.3,
                           "forest_fires":      0.3,
                           "panama":            0.3,
                           "wind_power":        0.1,
                           "india_ocean_waves": 0.3,
                           "szeged_weather":    0.25,
                           "turbine_power":     0.3,
                           "imu_gyro":          0.7,}

sliding_window_frac_dict = {"longterm_weather":  0.1, # ✅
                            "china_weather":     0.1, # ✅
                            "cali_housing":      0.25,
                            "bike_sharing":      0.25,
                            "forest_fires":      0.25,
                            "panama":            0.1, # ✅
                            "wind_power":        0.1, # ✅
                            "india_ocean_waves": 1/14, # ✅
                            "szeged_weather":    1/7, # ✅
                            "turbine_power":     1/7, # ✅
                            "imu_gyro":          0.25,}

angular_features_list = {"longterm_weather": [],
                        "electric_power":    [],
                        "china_weather":     [],
                        "cali_housing":      [],
                        "forest_fires":      [],
                        "panama":            [],
                        "wind_power":        [],
                        "imu_gyro":          [],
                        "bike_sharing":      ['season', 'mnth', 'hr', 'weekday'],
                        "india_ocean_waves": ['WDIR(degT)', 'MWD(degT)'],
                        "szeged_weather":    ['Wind Bearing (degrees)'],
                        "turbine_power":     ["winddirection_10m", "winddirection_100m"],
                        }

dataset_method_mapping = {"longterm_weather":  "periodic",
                          "china_weather":     "periodic",
                          "panama":            "periodic",
                          "wind_power":        "periodic",
                          "india_ocean_waves": "angular",
                          "szeged_weather":    "angular",
                          "turbine_power":     "angular",
                          "imu_gyro":          "?",}

