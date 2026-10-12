import numpy as np
import pandas as pd
CORE = ["PM2.5", "PM10", "SO2", "CO", "NO2", "NO", "O3"]

def pdk_renames(columns: list[str]) -> dict[str, str]:
    """Keep regulatory ratios separate from concentrations."""
    mapping = {}
    for col in columns:
        if col in {"PM2.5_2", "PM10_2", "O3_2"}:
            mapping[col] = col[:-2] + "_PDK_ratio"
        elif "/ПДК" in str(col):
            mapping[col] = str(col).replace("/ПДК", "_PDK_ratio")
    return mapping

def city_series(hourly: pd.DataFrame) -> pd.DataFrame:
    grouped = hourly.groupby("timestamp", sort=True)
    city = grouped[CORE].mean()
    for col in CORE:
        city[f"n_{col}"] = grouped[col].count()
    full_index = pd.date_range(city.index.min(), city.index.max(), freq="h", name="timestamp")
    city = city.reindex(full_index)
    city.index.name = "timestamp"
    city = city.reset_index()
    city["year"] = city.timestamp.dt.year
    city["month"] = city.timestamp.dt.month
    city["hour"] = city.timestamp.dt.hour
    city["day_of_week"] = city.timestamp.dt.dayofweek
    city["day_of_year"] = city.timestamp.dt.dayofyear
    city["hour_sin"] = np.sin(2 * np.pi * city.hour / 24)
    city["hour_cos"] = np.cos(2 * np.pi * city.hour / 24)
    city["doy_sin"] = np.sin(2 * np.pi * city.day_of_year / 365.25)
    city["doy_cos"] = np.cos(2 * np.pi * city.day_of_year / 365.25)
    city["dow_sin"] = np.sin(2 * np.pi * city.day_of_week / 7)
    city["dow_cos"] = np.cos(2 * np.pi * city.day_of_week / 7)
    return city

def add_lags(city: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = city.copy()
    feature_columns = []
    for lag in (0, 1, 3, 6):
        for pollutant in CORE:
            name = f"{pollutant}_lag{lag}"
            out[name] = out[pollutant].shift(lag)
            feature_columns.append(name)
    feature_columns += ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]
    out["target_NO2_next_hour"] = out["NO2"].shift(-1)
    out["target_time"] = out["timestamp"] + pd.Timedelta(hours=1)
    return out, feature_columns