from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Объединенный_датасет_Алматы_2026.xlsx"


POLLUTANT_ALIASES = {
    "so2": "SO2",
    "co": "CO",
    "no2": "NO2",
    "no": "NO",
    "o3": "O3",
    "о3": "O3",
    "озон": "O3",
    "диоксид серы": "SO2",
    "углерода оксид": "CO",
    "оксид углерода": "CO",
    "азота диоксид": "NO2",
    "диоксид азота": "NO2",
    "азота оксид": "NO",
    "оксид азота": "NO",
    "взвеш.в-ва": "Взвешенные вещества",
    "взвешенные вещества": "Взвешенные вещества",
    "фенол": "Фенол",
    "формальдегид": "Формальдегид",
    "бензол": "Бензол",
    "хлорбензол": "Хлорбензол",
    "этилбензол": "Этилбензол",
    "параксилол": "Параксилол",
    "метаксилол": "Метаксилол",
    "кумол": "Кумол",
    "ортаксилол": "Ортаксилол",
    "бенз(а)пирен": "Бенз(а)пирен",
    "кадмий": "Кадмий",
    "медь": "Медь",
    "мышьяк": "Мышьяк",
    "свинец": "Свинец",
    "хром (6+)": "Хром (6+)",
    "никель": "Никель",
    "цинк": "Цинк",
}


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def canonical_pollutant(value) -> str:
    text = clean_text(value)
    low = text.lower().replace("ё", "е")
    compact = re.sub(r"[\s_\-–—,.]", "", low)

    if "рм" in compact or "pm" in compact:
        if "25" in compact or "025" in compact or "02,5" in low:
            return "PM2.5"
        if "10" in compact or "010" in compact:
            return "PM10"
    if compact in {"so2", "co", "no2", "no", "o3"}:
        return compact.upper()
    return POLLUTANT_ALIASES.get(low, text)


def month_from_filename(path: Path) -> str:
    low = path.name.lower()
    for month in ("январь", "февраль", "март", "апрель", "май", "июнь"):
        if month in low:
            return month
    return ""


def numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.map(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
    return pd.to_numeric(series, errors="coerce")


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        result.append(name if count == 1 else f"{name}_{count}")
    return result


def parse_automatic(path: Path) -> list[pd.DataFrame]:
    result = []
    book = pd.ExcelFile(path)
    for sheet in book.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.empty:
            continue
        dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce", dayfirst=True)
        mask = dates.notna()
        if not mask.any():
            continue

        # Layout differs slightly between workbooks. Select the preceding row
        # containing the largest number of recognizable pollutant names.
        first_data_row = int(np.flatnonzero(mask.to_numpy())[0])
        known = {"PM2.5", "PM10", "SO2", "CO", "NO2", "NO", "O3"}
        candidates = range(max(0, first_data_row - 5), first_data_row)
        header_row = max(
            candidates,
            key=lambda row: sum(canonical_pollutant(raw.iat[row, col]) in known for col in range(1, raw.shape[1])),
        )
        original_dates = dates[mask].reset_index(drop=True)
        output = pd.DataFrame({
            "Дата_время": original_dates,
            "Исходная_дата_время": original_dates,
            "Дата_скорректирована": False,
        })
        names = []
        source_cols = []
        for col in range(1, raw.shape[1]):
            name = canonical_pollutant(raw.iat[header_row, col])
            if not name or name.lower() == "nan":
                continue
            names.append(name)
            source_cols.append(col)
        names = unique_names(names)
        for name, col in zip(names, source_cols):
            output[name] = numeric_series(raw.loc[mask, col]).reset_index(drop=True)

        station_address = ""
        if header_row > 0:
            candidates = [clean_text(raw.iat[r, c]) for r in range(header_row) for c in range(raw.shape[1])]
            station_address = max(candidates, key=len, default="")

        output.insert(0, "Адрес_или_описание", station_address)
        output.insert(0, "Пост", clean_text(sheet))
        output.insert(0, "Месяц_файла", month_from_filename(path))
        output.insert(0, "Тип_наблюдения", "автоматический")
        output.insert(0, "Исходный_файл", path.name)
        result.append(output)
    return result


def parse_manual(path: Path) -> list[pd.DataFrame]:
    result = []
    book = pd.ExcelFile(path)
    for sheet in book.sheet_names:
        if clean_text(sheet).lower() == "sysinf":
            continue
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.shape[0] < 12:
            continue
        dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
        mask = dates.notna()
        if not mask.any():
            continue

        first_data_row = int(np.flatnonzero(mask.to_numpy())[0])
        header_row = first_data_row - 1
        hour = numeric_series(raw.loc[mask, 1]).fillna(0)
        original_datetimes = dates[mask].reset_index(drop=True) + pd.to_timedelta(hour.reset_index(drop=True), unit="h")
        declared_year = pd.to_numeric(pd.Series([raw.iat[4, 1]]), errors="coerce").iloc[0]
        corrected_datetimes = original_datetimes.copy()
        correction = pd.Series(False, index=original_datetimes.index)
        if pd.notna(declared_year):
            correction = original_datetimes.dt.year.ne(int(declared_year))
            corrected_datetimes.loc[correction] = original_datetimes.loc[correction].map(
                lambda x: x.replace(year=int(declared_year))
            )
        output = pd.DataFrame({
            "Дата_время": corrected_datetimes,
            "Исходная_дата_время": original_datetimes,
            "Дата_скорректирована": correction,
            "Срок_наблюдения": hour.reset_index(drop=True).astype("Int64"),
        })

        names = []
        source_cols = []
        for col in range(2, raw.shape[1]):
            name = canonical_pollutant(raw.iat[header_row, col])
            if not name:
                continue
            names.append(name)
            source_cols.append(col)
        names = unique_names(names)
        for name, col in zip(names, source_cols):
            output[name] = numeric_series(raw.loc[mask, col]).reset_index(drop=True)

        output.insert(0, "Адрес_или_описание", "")
        output.insert(0, "Пост", clean_text(sheet))
        output.insert(0, "Месяц_файла", month_from_filename(path))
        output.insert(0, "Тип_наблюдения", "ручной/комбинированный")
        output.insert(0, "Исходный_файл", path.name)
        result.append(output)
    return result


def main() -> None:
    frames: list[pd.DataFrame] = []
    for path in sorted(ROOT.rglob("*.xls")):
        frames.extend(parse_automatic(path))
    for path in sorted(ROOT.rglob("*.xlsx")):
        if path.resolve() == OUTPUT.resolve():
            continue
        frames.extend(parse_manual(path))
    if not frames:
        raise RuntimeError("Не найдено наблюдений для объединения")

    data = pd.concat(frames, ignore_index=True, sort=False)
    # Align automatic observations with all sampling terms present in the
    # manual workbooks: 01:00, 07:00, 13:00 and 19:00.
    manual_terms = sorted(
        data.loc[data["Тип_наблюдения"].eq("ручной/комбинированный"), "Срок_наблюдения"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    is_automatic = data["Тип_наблюдения"].eq("автоматический")
    keep_automatic_time = (
        data["Дата_время"].dt.hour.isin(manual_terms)
        & data["Дата_время"].dt.minute.eq(0)
        & data["Дата_время"].dt.second.eq(0)
    )
    rows_before_time_alignment = len(data)
    data = data.loc[~is_automatic | keep_automatic_time].copy()
    removed_by_time_alignment = rows_before_time_alignment - len(data)
    fixed = ["Исходный_файл", "Тип_наблюдения", "Месяц_файла", "Пост", "Адрес_или_описание", "Дата_время", "Исходная_дата_время", "Дата_скорректирована", "Срок_наблюдения"]
    fixed = [c for c in fixed if c in data.columns]
    measurements = [c for c in data.columns if c not in fixed]
    # Remove measurement columns that are empty across the whole dataset,
    # then remove scheduled rows containing no measurements at all.
    measurements = [c for c in measurements if data[c].notna().any()]
    rows_before_cleanup = len(data)
    data = data.loc[data[measurements].notna().any(axis=1)].copy()
    removed_empty_rows = rows_before_cleanup - len(data)
    # Dataset convention confirmed by the owner: an empty measurement cell
    # in a retained observation means zero, not an unknown value.
    filled_measurement_cells = int(data[measurements].isna().sum().sum())
    data[measurements] = data[measurements].fillna(0)
    all_zero_measurements = data[measurements].eq(0).all(axis=1)
    removed_all_zero_rows = int(all_zero_measurements.sum())
    data = data.loc[~all_zero_measurements].copy()
    data = data[fixed + measurements].sort_values(["Дата_время", "Тип_наблюдения", "Пост"], kind="stable")

    summary = (
        data.groupby(["Тип_наблюдения", "Пост"], dropna=False)
        .agg(
            Строк=("Дата_время", "size"),
            Начало=("Дата_время", "min"),
            Конец=("Дата_время", "max"),
        )
        .reset_index()
    )

    readme = pd.DataFrame(
        {
            "Параметр": [
                "Назначение",
                "Период",
                "Гранулярность",
                "Единицы концентраций",
                "Пропуски",
                "Исходные файлы",
            ],
            "Описание": [
                "Единая таблица мониторинга атмосферного воздуха Алматы",
                f"{data['Дата_время'].min():%d.%m.%Y %H:%M} — {data['Дата_время'].max():%d.%m.%Y %H:%M}",
                "Одна строка соответствует одному сроку наблюдения на одном посту",
                "мг/м³, если иное не было указано в исходном файле",
                f"Автоматические данные оставлены только для ручных сроков {', '.join(f'{h:02d}:00' for h in manual_terms)}; исключено {removed_by_time_alignment} промежуточных строк. Удалено {removed_empty_rows} строк без единого измерения, {removed_all_zero_rows} строк со всеми нулевыми показателями и все полностью пустые столбцы. В остальных ячейках показателей пропуски заменены на 0 по правилу владельца данных ({filled_measurement_cells} замен до удаления полностью нулевых строк). Ошибочный 2025 год ПНЗ №16 за май исправлен по полю «Год=2026», исходная дата сохранена",
                str(data["Исходный_файл"].nunique()),
            ],
        }
    )

    with pd.ExcelWriter(OUTPUT, engine="openpyxl", datetime_format="DD.MM.YYYY HH:MM") as writer:
        data.to_excel(writer, sheet_name="Все наблюдения", index=False)
        summary.to_excel(writer, sheet_name="Сводка по постам", index=False)
        readme.to_excel(writer, sheet_name="Описание", index=False)
        for sheet_name in writer.book.sheetnames:
            ws = writer.book[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

    print(f"Создан файл: {OUTPUT}")
    print(f"Строк наблюдений: {len(data):,}")
    print(f"Удалено при приведении к ручным срокам: {removed_by_time_alignment:,}")
    print(f"Удалено пустых строк: {removed_empty_rows:,}")
    print(f"Удалено полностью нулевых строк: {removed_all_zero_rows:,}")
    print(f"Пустых ячеек показателей заменено на 0: {filled_measurement_cells:,}")
    print(f"Столбцов: {len(data.columns)}")
    print(f"Постов/обозначений: {data['Пост'].nunique()}")
    print(f"Период: {data['Дата_время'].min()} — {data['Дата_время'].max()}")


if __name__ == "__main__":
    main()
