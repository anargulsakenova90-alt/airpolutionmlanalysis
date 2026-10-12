from pathlib import Path
import re
import numpy as np
import pandas as pd

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
    for month in ("январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"):
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