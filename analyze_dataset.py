from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Объединенный_датасет_Алматы_2026.xlsx"
REPORT = ROOT / "Статистический_анализ_Алматы_2026.xlsx"
CHART = ROOT / "График_средних_по_месяцам.png"

MONTH_ORDER = ["январь", "февраль", "март", "апрель", "май", "июнь"]
MAIN_POLLUTANTS = ["PM2.5", "PM10", "SO2", "CO", "NO2", "NO", "O3"]


def main() -> None:
    data = pd.read_excel(DATASET, sheet_name="Все наблюдения")
    metadata = list(data.columns[:9])
    indicators = list(data.columns[9:])

    descriptive = data[indicators].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index()
    descriptive = descriptive.rename(
        columns={
            "index": "Показатель",
            "count": "Количество",
            "mean": "Среднее",
            "std": "Стандартное_отклонение",
            "min": "Минимум",
            "25%": "Квартиль_25%",
            "50%": "Медиана",
            "75%": "Квартиль_75%",
            "max": "Максимум",
        }
    )
    descriptive["Нулевых_значений"] = [int(data[c].eq(0).sum()) for c in indicators]
    descriptive["Доля_нулей_%"] = [round(data[c].eq(0).mean() * 100, 2) for c in indicators]

    monthly = (
        data.assign(
            Месяц=pd.Categorical(data["Месяц_файла"].str.lower(), MONTH_ORDER, ordered=True)
        )
        .groupby("Месяц", observed=True)[indicators]
        .agg(["count", "mean", "median", "min", "max", "std"])
    )
    monthly.columns = [f"{indicator}_{stat}" for indicator, stat in monthly.columns]
    monthly = monthly.reset_index()

    by_station = (
        data.groupby(["Тип_наблюдения", "Пост"])[indicators]
        .agg(["count", "mean", "median", "min", "max"])
    )
    by_station.columns = [f"{indicator}_{stat}" for indicator, stat in by_station.columns]
    by_station = by_station.reset_index()

    zero_rows = data.loc[data[indicators].eq(0).all(axis=1), metadata].copy()
    overview = pd.DataFrame(
        {
            "Параметр": [
                "Количество строк",
                "Количество показателей",
                "Начало периода",
                "Конец периода",
                "Автоматических строк",
                "Ручных/комбинированных строк",
                "Полностью нулевых строк",
                "Доля полностью нулевых строк, %",
            ],
            "Значение": [
                len(data),
                len(indicators),
                data["Дата_время"].min(),
                data["Дата_время"].max(),
                int(data["Тип_наблюдения"].eq("автоматический").sum()),
                int(data["Тип_наблюдения"].eq("ручной/комбинированный").sum()),
                len(zero_rows),
                round(len(zero_rows) / len(data) * 100, 2),
            ],
        }
    )

    with pd.ExcelWriter(REPORT, engine="openpyxl", datetime_format="DD.MM.YYYY HH:MM") as writer:
        overview.to_excel(writer, sheet_name="Общая сводка", index=False)
        descriptive.to_excel(writer, sheet_name="Описательная статистика", index=False)
        monthly.to_excel(writer, sheet_name="По месяцам", index=False)
        by_station.to_excel(writer, sheet_name="По постам", index=False)
        zero_rows.to_excel(writer, sheet_name="Полностью нулевые строки", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

    monthly_means = (
        data.assign(
            Месяц=pd.Categorical(data["Месяц_файла"].str.lower(), MONTH_ORDER, ordered=True)
        )
        .groupby("Месяц", observed=True)[MAIN_POLLUTANTS]
        .mean()
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    axes = axes.ravel()
    colors = ["#2878B5", "#55A868", "#C44E52", "#DD8452", "#8172B2", "#937860", "#4C9F70"]
    for ax, pollutant, color in zip(axes, MAIN_POLLUTANTS, colors):
        values = monthly_means[pollutant]
        ax.plot(values.index.astype(str), values.values, marker="o", linewidth=2.2, color=color)
        ax.fill_between(range(len(values)), values.values, alpha=0.12, color=color)
        ax.set_title(pollutant, fontweight="bold")
        ax.set_ylabel("Средняя концентрация, мг/м³")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
        for i, value in enumerate(values.values):
            ax.annotate(f"{value:.4f}", (i, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axes[-1].axis("off")
    fig.suptitle(
        "Алматы: средние концентрации загрязнителей по месяцам, 2026\n"
        "Сроки наблюдений: 01:00, 07:00, 13:00 и 19:00; нулевые строки включены",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(CHART, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Отчёт: {REPORT}")
    print(f"График: {CHART}")
    print(f"Строк: {len(data)}; показателей: {len(indicators)}; полностью нулевых строк: {len(zero_rows)}")
    print(descriptive[["Показатель", "Среднее", "Медиана", "Минимум", "Максимум", "Доля_нулей_%"]].to_string(index=False))


if __name__ == "__main__":
    main()
