from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "модели"
MODEL_DIR.mkdir(exist_ok=True)
INPUTS = sorted(ROOT.glob("Датасет_импутация_*.xlsx"))
POLLUTANTS = ["PM2.5", "PM10", "SO2", "CO", "NO2", "NO", "O3"]
RANDOM_STATE = 42


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", value).strip("_")


def features(data: pd.DataFrame, include_pollutants: list[str]) -> pd.DataFrame:
    result = data[include_pollutants].copy()
    result["месяц"] = data["Дата_время"].dt.month
    result["день_недели"] = data["Дата_время"].dt.dayofweek
    result["час"] = data["Дата_время"].dt.hour
    station = pd.get_dummies(data["Пост"], prefix="пост", dtype=float)
    result = pd.concat([result.reset_index(drop=True), station.reset_index(drop=True)], axis=1)
    for col in result.columns:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())
    return result.astype(float)


def chronological_split(data: pd.DataFrame, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    order = data["Дата_время"].sort_values(kind="stable").index.to_numpy()
    cut = max(1, int(len(order) * ratio))
    return order[:cut], order[cut:]


def regression_metrics(y_true, y_pred, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Модель": [label],
            "MAE": [mean_absolute_error(y_true, y_pred)],
            "RMSE": [root_mean_squared_error(y_true, y_pred)],
            "R2": [r2_score(y_true, y_pred)],
            "Тестовых_строк": [len(y_true)],
        }
    )


def run_one(path: Path) -> None:
    data = pd.read_excel(path, sheet_name="Все наблюдения")
    data["Дата_время"] = pd.to_datetime(data["Дата_время"])
    tag = safe_name(path.stem.replace("Датасет_импутация_", ""))
    report = ROOT / f"Результаты_6_моделей_{tag}.xlsx"

    # 1. One-step forecast of NO2 within each station.
    forecast = data.sort_values(["Пост", "Дата_время"]).copy()
    forecast["Цель_NO2_следующий_срок"] = forecast.groupby("Пост")["NO2"].shift(-1)
    forecast = forecast.dropna(subset=["Цель_NO2_следующий_срок"]).reset_index(drop=True)
    x_forecast = features(forecast, POLLUTANTS)
    y_forecast = forecast["Цель_NO2_следующий_срок"].astype(float)
    train, test = chronological_split(forecast)
    forecast_model = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    )
    forecast_model.fit(x_forecast.loc[train], y_forecast.loc[train])
    forecast_pred = forecast_model.predict(x_forecast.loc[test])
    forecast_metrics = regression_metrics(y_forecast.loc[test], forecast_pred, "Прогноз NO2 на следующий срок")
    forecast_output = forecast.loc[test, ["Дата_время", "Пост", "Цель_NO2_следующий_срок"]].copy()
    forecast_output["Прогноз_NO2"] = forecast_pred
    forecast_output["Абсолютная_ошибка"] = np.abs(
        forecast_output["Цель_NO2_следующий_срок"] - forecast_output["Прогноз_NO2"]
    )
    joblib.dump(
        {"model": forecast_model, "features": list(x_forecast.columns), "target": "NO2 next term"},
        MODEL_DIR / f"{tag}_01_прогноз_NO2.joblib",
    )

    # 2. Anomaly detection.
    anomaly_x = data[POLLUTANTS].copy()
    anomaly_x = anomaly_x.fillna(anomaly_x.median())
    anomaly_scaler = StandardScaler()
    anomaly_scaled = anomaly_scaler.fit_transform(anomaly_x)
    anomaly_model = IsolationForest(contamination=0.03, random_state=RANDOM_STATE, n_jobs=-1)
    anomaly_label = anomaly_model.fit_predict(anomaly_scaled)
    anomaly_score = -anomaly_model.score_samples(anomaly_scaled)
    anomaly_output = data[["Дата_время", "Пост", "Тип_наблюдения"] + POLLUTANTS].copy()
    anomaly_output["Аномалия"] = anomaly_label == -1
    anomaly_output["Оценка_аномальности"] = anomaly_score
    anomaly_output = anomaly_output.sort_values("Оценка_аномальности", ascending=False)
    joblib.dump(
        {"model": anomaly_model, "scaler": anomaly_scaler, "features": POLLUTANTS},
        MODEL_DIR / f"{tag}_02_IsolationForest.joblib",
    )

    # 3. Clustering of stations by average pollution profile.
    station_profile = data.groupby(["Тип_наблюдения", "Пост"])[POLLUTANTS].mean()
    station_profile = station_profile.fillna(station_profile.median()).fillna(0)
    cluster_scaler = StandardScaler()
    station_scaled = cluster_scaler.fit_transform(station_profile)
    cluster_model = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
    clusters = cluster_model.fit_predict(station_scaled)
    cluster_output = station_profile.reset_index()
    cluster_output["Кластер"] = clusters + 1
    joblib.dump(
        {"model": cluster_model, "scaler": cluster_scaler, "features": POLLUTANTS},
        MODEL_DIR / f"{tag}_03_KMeans.joblib",
    )

    # 4. Classification of high NO2 (upper quartile) without using current NO2 as a predictor.
    class_data = data.dropna(subset=["NO2"]).reset_index(drop=True)
    threshold = float(class_data["NO2"].quantile(0.75))
    class_y = class_data["NO2"].gt(threshold).astype(int)
    class_x = features(class_data, [p for p in POLLUTANTS if p != "NO2"])
    train, test = chronological_split(class_data)
    class_model = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    class_model.fit(class_x.loc[train], class_y.loc[train])
    class_pred = class_model.predict(class_x.loc[test])
    class_prob = class_model.predict_proba(class_x.loc[test])[:, 1]
    class_metrics = pd.DataFrame(
        {
            "Порог_NO2": [threshold],
            "Accuracy": [accuracy_score(class_y.loc[test], class_pred)],
            "Precision": [precision_score(class_y.loc[test], class_pred, zero_division=0)],
            "Recall": [recall_score(class_y.loc[test], class_pred, zero_division=0)],
            "F1": [f1_score(class_y.loc[test], class_pred, zero_division=0)],
            "ROC_AUC": [roc_auc_score(class_y.loc[test], class_prob)],
            "Тестовых_строк": [len(test)],
        }
    )
    class_output = class_data.loc[test, ["Дата_время", "Пост", "NO2"]].copy()
    class_output["Фактически_высокий_NO2"] = class_y.loc[test].to_numpy()
    class_output["Прогноз_класса"] = class_pred
    class_output["Вероятность"] = class_prob
    joblib.dump(
        {"model": class_model, "features": list(class_x.columns), "threshold": threshold},
        MODEL_DIR / f"{tag}_04_классификация_NO2.joblib",
    )

    # 5. Factor analysis via PCA.
    pca_x = data[POLLUTANTS].fillna(data[POLLUTANTS].median())
    pca_scaler = StandardScaler()
    pca_scaled = pca_scaler.fit_transform(pca_x)
    pca_model = PCA(n_components=3, random_state=RANDOM_STATE)
    scores = pca_model.fit_transform(pca_scaled)
    pca_loadings = pd.DataFrame(
        pca_model.components_.T,
        index=POLLUTANTS,
        columns=["Компонента_1", "Компонента_2", "Компонента_3"],
    ).reset_index(names="Показатель")
    pca_variance = pd.DataFrame(
        {
            "Компонента": [1, 2, 3],
            "Объяснённая_дисперсия_%": pca_model.explained_variance_ratio_ * 100,
            "Накопленная_дисперсия_%": np.cumsum(pca_model.explained_variance_ratio_) * 100,
        }
    )
    joblib.dump(
        {"model": pca_model, "scaler": pca_scaler, "features": POLLUTANTS},
        MODEL_DIR / f"{tag}_05_PCA.joblib",
    )

    # 6. Station-temporal model: NO2 from station and calendar only.
    st_data = data.dropna(subset=["NO2"]).reset_index(drop=True)
    st_x = features(st_data, [])
    st_y = st_data["NO2"].astype(float)
    train, test = chronological_split(st_data)
    st_model = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
    )
    st_model.fit(st_x.loc[train], st_y.loc[train])
    st_pred = st_model.predict(st_x.loc[test])
    st_metrics = regression_metrics(st_y.loc[test], st_pred, "Пост-время → NO2")
    st_output = st_data.loc[test, ["Дата_время", "Пост", "NO2"]].copy()
    st_output["Прогноз_NO2"] = st_pred
    st_output["Абсолютная_ошибка"] = np.abs(st_output["NO2"] - st_output["Прогноз_NO2"])
    joblib.dump(
        {"model": st_model, "features": list(st_x.columns), "note": "station id used; coordinates unavailable"},
        MODEL_DIR / f"{tag}_06_пост_время_NO2.joblib",
    )

    summary = pd.DataFrame(
        {
            "Тип": [
                "1. Прогнозирование", "2. Аномалии", "3. Кластеризация",
                "4. Классификация", "5. PCA", "6. Пост-время",
            ],
            "Метод": [
                "Random Forest: NO2 следующего срока", "Isolation Forest, 3%",
                "K-means, 4 кластера", "Random Forest: верхний квартиль NO2",
                "PCA, 3 компоненты", "Random Forest без координат",
            ],
            "Основной_результат": [
                f"R2={forecast_metrics.at[0, 'R2']:.4f}; RMSE={forecast_metrics.at[0, 'RMSE']:.4f}",
                f"Аномалий={(anomaly_label == -1).sum()}",
                f"Постов={len(cluster_output)}",
                f"F1={class_metrics.at[0, 'F1']:.4f}; AUC={class_metrics.at[0, 'ROC_AUC']:.4f}",
                f"Дисперсия 3 компонент={pca_variance['Объяснённая_дисперсия_%'].sum():.2f}%",
                f"R2={st_metrics.at[0, 'R2']:.4f}; RMSE={st_metrics.at[0, 'RMSE']:.4f}",
            ],
        }
    )

    with pd.ExcelWriter(report, engine="openpyxl", datetime_format="DD.MM.YYYY HH:MM") as writer:
        summary.to_excel(writer, sheet_name="Сводка", index=False)
        forecast_metrics.to_excel(writer, sheet_name="1_Прогноз_метрики", index=False)
        forecast_output.to_excel(writer, sheet_name="1_Прогнозы", index=False)
        anomaly_output.to_excel(writer, sheet_name="2_Аномалии", index=False)
        cluster_output.to_excel(writer, sheet_name="3_Кластеры", index=False)
        class_metrics.to_excel(writer, sheet_name="4_Классиф_метрики", index=False)
        class_output.to_excel(writer, sheet_name="4_Классификация", index=False)
        pca_variance.to_excel(writer, sheet_name="5_PCA_дисперсия", index=False)
        pca_loadings.to_excel(writer, sheet_name="5_PCA_нагрузки", index=False)
        st_metrics.to_excel(writer, sheet_name="6_Пост_время_метрики", index=False)
        st_output.to_excel(writer, sheet_name="6_Пост_время", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
    print(f"{path.name} -> {report.name}")
    print(summary.to_string(index=False))


def main() -> None:
    if not INPUTS:
        raise FileNotFoundError("Не найдены датасеты с импутацией")
    for path in INPUTS:
        run_one(path)


if __name__ == "__main__":
    main()
