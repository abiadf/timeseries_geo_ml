"""Dictionaties for datasets to use in main code"""

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
                "imu_gyro":          [],
                "gas":               [],
                "beijing":           ["No","year","month","day","hour", "station"],}

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
                           "imu_gyro":          0.7,
                           "gas":               0.3,
                           "beijing":           0.3,}

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
                            "imu_gyro":          0.25,
                            "gas":               1/7,
                            "beijing":           1/7,}

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
                        "gas":               [],
                        "beijing":           ["wd_deg"],}

dataset_method_mapping = {"longterm_weather":  "periodic",
                          "china_weather":     "periodic",
                          "panama":            "periodic",
                          "wind_power":        "periodic",
                          "india_ocean_waves": "angular",
                          "szeged_weather":    "angular",
                          "turbine_power":     "angular",
                          "imu_gyro":          "?",}

dataset_attributes = {
                      "beijing":           {"cols_to_drop":        [],
                                            "mapping":             "?",
                                            "angular_features":    ["wd_deg"], # 📐
                                            "sliding_window_frac": 1/7,
                                            "periodic_threshold":  0.3},

                      "china_weather":     {"cols_to_drop":        [],
                                            "mapping":             "periodic",
                                            "angular_features":    [], # 
                                            "sliding_window_frac": 0.1,
                                            "periodic_threshold":  0.3},

                      "gas":               {"cols_to_drop":        [],
                                            "mapping":             "?",
                                            "angular_features":    [], # 
                                            "sliding_window_frac": 1/7,
                                            "periodic_threshold":  0.3},

                      "imu_gyro":          {"cols_to_drop":        [],
                                            "mapping":             "?",
                                            "angular_features":    [], # 
                                            "sliding_window_frac": 0.25,
                                            "periodic_threshold":  0.7},

                      "india_ocean_waves": {"cols_to_drop":        ['ID', '#YY', 'MM', 'DD', 'hh', 'mm'],
                                            "mapping":             "angular",
                                            "angular_features":    ['WDIR(degT)', 'MWD(degT)'], # ⭐️ 🚴📐
                                            "sliding_window_frac": 1/28,
                                            "periodic_threshold":  0.3},

                      "longterm_weather":  {"cols_to_drop":        ["date"],
                                            "mapping":             "periodic",
                                            "angular_features":    [], # 
                                            "sliding_window_frac": 0.1,
                                            "periodic_threshold":  0.52},

                      "nasa_moon":         {"cols_to_drop":        ["date"],
                                            "mapping":             "periodic",
                                            "angular_features":    ['R.A._(ICRF)_sin', 'R.A._(ICRF)_cos', 'ObsEcLon_sin', 'ObsEcLon_cos', 'GlxLon_sin', 'GlxLon_cos','DEC__(ICRF)', 'S-T-O', 'ObsEcLat', 'GlxLat'],
                                            "sliding_window_frac": 1/10,
                                            "periodic_threshold":  0.52},  # ????

                      "panama":            {"cols_to_drop":        ["datetime"],
                                            "mapping":             "periodic",
                                            "angular_features":    [], # 
                                            "sliding_window_frac": 0.1,
                                            "periodic_threshold":  0.3},

                      "szeged_weather":    {"cols_to_drop":        ["Precip Type","Summary", "Formatted Date", "Apparent Temperature (C)", "Loud Cover", "Daily Summary"],
                                            "mapping":             "angular",
                                            "angular_features":    ['Wind Bearing (degrees)'], #  📐
                                            "sliding_window_frac": 1/7,
                                            "periodic_threshold":  0.25},

                      "turbine_power":     {"cols_to_drop":        ["Time"],
                                            "mapping":             "angular",
                                            "angular_features":    ["winddirection_10m", "winddirection_100m"], # 📐
                                            "sliding_window_frac": 1/7,
                                            "periodic_threshold":  0.3},

                      "wind_power":        {"cols_to_drop":        ["Date/Time"],
                                            "mapping":             "periodic",
                                            "angular_features":    [], # 
                                            "sliding_window_frac": 0.1,
                                            "periodic_threshold":  0.1},
                    }

