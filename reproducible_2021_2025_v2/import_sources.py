from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd


from paths import OUTPUT as ROOT, SOURCE
DATA_DIR = ROOT / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
from source_parsers import canonical_pollutant, clean_text, numeric_series, parse_automatic

DATE_COL = "timestamp"
CORE = ["PM2.5", "PM10", "SO2", "CO", "NO2", "NO", "O3"]


def metadata_values(raw: pd.DataFrame) -> dict[str, str]:
    output = {}
    labels = {"год": "declared_year", "месяц": "declared_month"}
    for _, row in raw.iloc[:12, :2].iterrows():
        label = clean_text(row.iloc[0]).lower().replace("ё", "е")
        if label in labels and len(row) > 1 and pd.notna(row.iloc[1]):
            output[labels[label]] = clean_text(row.iloc[1])
    return output


def parse_manual(path: Path, folder_year: int) -> tuple[pd.DataFrame, list[dict]]:
    records = []
    conflicts = []
    book = pd.ExcelFile(path)
    for sheet in book.sheet_names:
        if clean_text(sheet).lower() in {"sysinf", "дополн. информация", "дополн.информация"}:
            continue
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.shape[0] < 12 or raw.shape[1] < 3:
            continue
        dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce", dayfirst=True, format="mixed")
        mask = dates.notna()
        if not mask.any():
            continue
        first_data = int(np.flatnonzero(mask.to_numpy())[0])
        header_row = first_data - 1
        if header_row < 0:
            continue
        meta = metadata_values(raw)
        hour_raw = numeric_series(raw.loc[mask, 1]).reset_index(drop=True)
        date_values = dates.loc[mask].reset_index(drop=True)
        hours = hour_raw.fillna(date_values.dt.hour).fillna(0)
        base_dates = date_values.where(date_values.dt.hour.eq(0), date_values.dt.normalize())
        timestamps = base_dates + pd.to_timedelta(hours, unit="h")
        fields: dict[str, pd.Series] = {}
        for col in range(2, raw.shape[1]):
            name = canonical_pollutant(raw.iat[header_row, col])
            if not name or name.lower() == "nan":
                continue
            values = numeric_series(raw.loc[mask, col]).reset_index(drop=True)
            if name in fields:
                suffix = 2
                while f"{name}_{suffix}" in fields:
                    suffix += 1
                name = f"{name}_{suffix}"
            fields[name] = values
        if not fields:
            continue
        item = pd.DataFrame(fields)
        item.insert(0, "timestamp", timestamps)
        item["source_type"] = "manual"
        item["station_raw"] = clean_text(sheet)
        item["station_key"] = "manual|" + re.sub(r"\s+", " ", clean_text(sheet).lower())
        item["source_file"] = str(path.relative_to(SOURCE))
        item["folder_year"] = folder_year
        item["declared_year"] = meta.get("declared_year", "")
        item["declared_month"] = meta.get("declared_month", "")
        item["actual_year"] = item.timestamp.dt.year
        item["actual_month"] = item.timestamp.dt.month
        item["sampling_hour"] = hours.astype("Int64")
        item["metadata_mismatch"] = (
            (item.timestamp.dt.year.astype(str) != str(meta.get("declared_year", "")))
            if meta.get("declared_year") else False
        )
        item["all_reported_zero"] = item[list(fields)].notna().any(axis=1) & item[list(fields)].fillna(0).eq(0).all(axis=1)
        years = sorted(item.timestamp.dt.year.unique().tolist())
        if meta.get("declared_year") and years != [int(float(meta["declared_year"]))]:
            conflicts.append({
                "file": str(path.relative_to(SOURCE)), "sheet": clean_text(sheet),
                "folder_year": folder_year, "declared_year": meta.get("declared_year"),
                "actual_years": ",".join(map(str, years)), "action": "kept actual observation dates; no blind year rewrite",
            })
        records.append(item)
    if not records:
        return pd.DataFrame(), conflicts
    return pd.concat(records, ignore_index=True, sort=False), conflicts


def normalize_auto(path: Path, folder_year: int) -> pd.DataFrame:
    records = parse_automatic(path)
    parts = []
    for frame in records:
        if frame.empty:
            continue
        metadata = {"Дата_время", "Исходная_дата_время", "Дата_скорректирована", "Пост", "Тип_наблюдения", "Исходный_файл", "Месяц_файла", "Адрес_или_описание"}
        measurements = [c for c in frame.columns if c not in metadata]
        out = frame[measurements].copy()
        out.insert(0, "timestamp", pd.to_datetime(frame["Дата_время"], errors="coerce"))
        out["source_type"] = "automatic"
        out["station_raw"] = frame["Пост"].astype(str).str.strip()
        out["station_key"] = "automatic|" + out.station_raw.str.lower().str.replace(r"\s+", " ", regex=True)
        out["source_file"] = str(path.relative_to(SOURCE))
        out["folder_year"] = folder_year
        out["declared_year"] = folder_year
        out["declared_month"] = ""
        out["actual_year"] = out.timestamp.dt.year
        out["actual_month"] = out.timestamp.dt.month
        out["sampling_hour"] = out.timestamp.dt.hour.astype("Int64")
        out["metadata_mismatch"] = out.timestamp.dt.year.ne(folder_year)
        out["all_reported_zero"] = out[measurements].notna().any(axis=1) & out[measurements].fillna(0).eq(0).all(axis=1)
        parts.append(out)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def write_partition(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, encoding="utf-8", compression="gzip", date_format="%Y-%m-%d %H:%M:%S")


def main() -> None:
    summaries = []
    conflicts = []
    parser_errors = []
    year_partitions = []
    hourly_parts = []
    for year in range(2021, 2026):
        year_root = SOURCE / str(year)
        xls_files = sorted(year_root.rglob("*.xls"))
        xlsx_files = sorted(year_root.rglob("*.xlsx"))
        chunks = []
        for path in xls_files:
            try:
                chunk = normalize_auto(path, year)
                if not chunk.empty:
                    chunks.append(chunk)
                summaries.append({"year_folder": year, "type": "automatic", "file": str(path.relative_to(SOURCE)), "parsed_rows": len(chunk), "sheets_or_stations": chunk.station_key.nunique() if not chunk.empty else 0, "status": "parsed" if not chunk.empty else "no parseable measurements"})
            except Exception as exc:
                parser_errors.append({"year_folder": year, "file": str(path.relative_to(SOURCE)), "error": repr(exc)})
                summaries.append({"year_folder": year, "type": "automatic", "file": str(path.relative_to(SOURCE)), "parsed_rows": 0, "sheets_or_stations": 0, "status": "ERROR"})
        for path in xlsx_files:
            try:
                chunk, file_conflicts = parse_manual(path, year)
                conflicts.extend(file_conflicts)
                if not chunk.empty:
                    chunks.append(chunk)
                summaries.append({"year_folder": year, "type": "manual", "file": str(path.relative_to(SOURCE)), "parsed_rows": len(chunk), "sheets_or_stations": chunk.station_key.nunique() if not chunk.empty else 0, "status": "parsed" if not chunk.empty else "no parseable measurements"})
            except Exception as exc:
                parser_errors.append({"year_folder": year, "file": str(path.relative_to(SOURCE)), "error": repr(exc)})
                summaries.append({"year_folder": year, "type": "manual", "file": str(path.relative_to(SOURCE)), "parsed_rows": 0, "sheets_or_stations": 0, "status": "ERROR"})
        if not chunks:
            raise RuntimeError(f"No observations parsed for {year}")
        year_data = pd.concat(chunks, ignore_index=True, sort=False)
        year_data = year_data.loc[year_data.timestamp.notna()].copy()
        # Preserve explicit zeros and missing cells separately; only truly empty rows are removed.
        measurement_cols = [c for c in year_data.columns if c not in {
            "timestamp", "source_type", "station_raw", "station_key", "source_file", "folder_year",
            "declared_year", "declared_month", "actual_year", "actual_month", "sampling_hour",
            "metadata_mismatch", "all_reported_zero",
        }]
        has_measurement = year_data[measurement_cols].notna().any(axis=1)
        year_data = year_data.loc[has_measurement].copy()
        year_data["month"] = year_data.timestamp.dt.month
        year_data["year"] = year_data.timestamp.dt.year
        partition = DATA_DIR / f"raw_observations_{year}.csv.gz"
        write_partition(year_data, partition)
        year_partitions.append(partition)
        automatic = year_data[year_data.source_type.eq("automatic")].copy()
        hourly = (
            automatic.set_index("timestamp")
            .groupby("station_key", sort=False)[measurement_cols]
            .resample("1h").mean()
            .reset_index()
        )
        station_names = automatic.groupby("station_key", sort=False).station_raw.first()
        hourly["station_raw"] = hourly.station_key.map(station_names)
        hourly["source_type"] = "automatic"
        hourly["year"] = hourly.timestamp.dt.year
        hourly["month"] = hourly.timestamp.dt.month
        hourly["n_station_values"] = hourly[measurement_cols].notna().sum(axis=1)
        hpath = DATA_DIR / f"automatic_hourly_{year}.csv.gz"
        write_partition(hourly, hpath)
        hourly_parts.append(hourly)
        summaries.append({"year_folder": year, "type": "combined", "file": partition.name, "parsed_rows": len(year_data), "sheets_or_stations": year_data.station_key.nunique(), "status": "saved"})
        print(f"{year}: raw rows={len(year_data):,}; automatic hourly rows={len(hourly):,}; sources={year_data.source_type.value_counts().to_dict()}", flush=True)

    hourly_all = pd.concat(hourly_parts, ignore_index=True, sort=False)
    hourly_measurements = [c for c in hourly_all.columns if c not in {"timestamp", "station_key", "station_raw", "source_type", "year", "month", "n_station_values"}]
    hourly_all = hourly_all.groupby(["station_key", "timestamp"], as_index=False)[hourly_measurements].mean()
    station_names = pd.concat(hourly_parts, ignore_index=True).groupby("station_key").station_raw.first()
    hourly_all["station_raw"] = hourly_all.station_key.map(station_names)
    hourly_all["source_type"] = "automatic"
    hourly_all["year"] = hourly_all.timestamp.dt.year
    hourly_all["month"] = hourly_all.timestamp.dt.month
    write_partition(hourly_all, DATA_DIR / "automatic_hourly_station_panel.csv.gz")
    pd.DataFrame(summaries).to_csv(ROOT / "source_file_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conflicts).to_csv(ROOT / "manual_metadata_conflicts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(parser_errors).to_csv(ROOT / "parser_errors.csv", index=False, encoding="utf-8-sig")
    overview = {
        "period": [str(hourly_all.timestamp.min()), str(hourly_all.timestamp.max())],
        "years": sorted(hourly_all.timestamp.dt.year.unique().tolist()),
        "source_files": len(summaries) - 5,
        "raw_partitions": [p.name for p in year_partitions],
        "raw_rows_by_year": {str(x['year_folder']): int(x['parsed_rows']) for x in summaries if x['type'] == 'combined'},
        "hourly_rows": int(len(hourly_all)),
        "automatic_station_keys": int(hourly_all.station_key.nunique()),
        "manual_source_files_2025": int(sum(x["year_folder"] == 2025 and x["type"] == "manual" for x in summaries)),
        "note": "2025 input folder supplied automatic .xls only; manual .xlsx files were absent. Actual timestamps were retained when workbook metadata conflicted.",
    }
    (ROOT / "dataset_manifest.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved hourly panel: rows={len(hourly_all):,}, stations={hourly_all.station_key.nunique()}, span={overview['period']}", flush=True)
    print(f"Parser errors={len(parser_errors)}; manual metadata conflicts={len(conflicts)}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Could not infer format")
    main()
