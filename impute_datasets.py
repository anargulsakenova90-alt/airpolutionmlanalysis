from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from combine_datasets import parse_automatic, parse_manual


ROOT = Path(__file__).resolve().parent
OUTPUTS = {
    "Среднее по посту": ROOT / "Датасет_импутация_средним.xlsx",
    "Медиана по посту": ROOT / "Датасет_импутация_медианой.xlsx",
    "Линейная интерполяция": ROOT / "Датасет_импутация_интерполяцией.xlsx",
    "KNN (5 соседей)": ROOT / "Датасет_импутация_KNN.xlsx",
}


def build_unfilled_dataset() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    frames: list[pd.DataFrame] = []
    for path in sorted(ROOT.rglob("*.xls")):
        frames.extend(parse_automatic(path))
    for path in sorted(ROOT.rglob("*.xlsx")):
        # Only original manual workbooks are inside the source subdirectory.
        if path.parent == ROOT:
            continue
        frames.extend(parse_manual(path))

    data = pd.concat(frames, ignore_index=True, sort=False)
    manual_terms = sorted(
        data.loc[data["Тип_наблюдения"].eq("ручной/комбинированный"), "Срок_наблюдения"]
        .dropna().astype(int).unique().tolist()
    )
    is_auto = data["Тип_наблюдения"].eq("автоматический")
    aligned = (
        data["Дата_время"].dt.hour.isin(manual_terms)
        & data["Дата_время"].dt.minute.eq(0)
        & data["Дата_время"].dt.second.eq(0)
    )
    before_time = len(data)
    data = data.loc[~is_auto | aligned].copy()

    metadata = [
        "Исходный_файл", "Тип_наблюдения", "Месяц_файла", "Пост",
        "Адрес_или_описание", "Дата_время", "Исходная_дата_время",
        "Дата_скорректирована", "Срок_наблюдения",
    ]
    measurements = [c for c in data.columns if c not in metadata and data[c].notna().any()]

    empty = data[measurements].isna().all(axis=1)
    removed_empty = int(empty.sum())
    data = data.loc[~empty].copy()

    # A row is fully zero if every explicitly present measurement is zero;
    # missing cells do not turn it into a valid nonzero observation.
    any_nonzero = data[measurements].fillna(0).ne(0).any(axis=1)
    removed_zero = int((~any_nonzero).sum())
    data = data.loc[any_nonzero, metadata + measurements].copy()
    data = data.sort_values(["Дата_время", "Тип_наблюдения", "Пост"], kind="stable").reset_index(drop=True)

    stats = {
        "Удалено промежуточных времён": before_time - (len(data) + removed_empty + removed_zero),
        "Удалено полностью пустых строк": removed_empty,
        "Удалено полностью нулевых строк": removed_zero,
    }
    return data, measurements, stats


def impute_group(group: pd.DataFrame, measurements: list[str], method: str) -> pd.DataFrame:
    group = group.sort_values("Дата_время").copy()
    applicable = [c for c in measurements if group[c].notna().any()]
    if not applicable:
        return group

    if method == "Среднее по посту":
        group[applicable] = group[applicable].fillna(group[applicable].mean())
    elif method == "Медиана по посту":
        group[applicable] = group[applicable].fillna(group[applicable].median())
    elif method == "Линейная интерполяция":
        group[applicable] = group[applicable].interpolate(method="linear", limit_direction="both")
    elif method == "KNN (5 соседей)":
        values = KNNImputer(n_neighbors=5, weights="distance").fit_transform(group[applicable])
        group.loc[:, applicable] = values
    else:
        raise ValueError(method)
    return group


def write_variant(
    base: pd.DataFrame,
    measurements: list[str],
    method: str,
    output: Path,
    cleanup_stats: dict[str, int],
) -> dict[str, int]:
    original_missing = base[measurements].isna()
    parts = []
    for _, group in base.groupby(["Тип_наблюдения", "Пост"], sort=False):
        parts.append(impute_group(group, measurements, method))
    result = pd.concat(parts).sort_index()

    # Count only imputations in applicable station/pollutant combinations.
    filled_mask = original_missing & result[measurements].notna()
    result.insert(9, "Количество_импутированных_значений", filled_mask.sum(axis=1).astype(int))
    imputed_count = int(filled_mask.sum().sum())
    structural_missing = int(result[measurements].isna().sum().sum())
    all_zero_rows = int(result[measurements].fillna(0).eq(0).all(axis=1).sum())

    description = pd.DataFrame(
        {
            "Параметр": [
                "Метод",
                "Строк",
                "Импутировано ячеек",
                "Структурных пропусков",
                "Полностью нулевых строк",
                "Правило структурных пропусков",
                *cleanup_stats.keys(),
            ],
            "Значение": [
                method,
                len(result),
                imputed_count,
                structural_missing,
                all_zero_rows,
                "Оставлены пустыми, если показатель никогда не измерялся на данном посту",
                *cleanup_stats.values(),
            ],
        }
    )
    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="DD.MM.YYYY HH:MM") as writer:
        result.to_excel(writer, sheet_name="Все наблюдения", index=False)
        description.to_excel(writer, sheet_name="Описание", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
    return {
        "rows": len(result),
        "imputed": imputed_count,
        "structural": structural_missing,
        "all_zero": all_zero_rows,
    }


def main() -> None:
    base, measurements, cleanup_stats = build_unfilled_dataset()
    print(f"Базовых строк: {len(base):,}; показателей: {len(measurements)}")
    for method, output in OUTPUTS.items():
        stats = write_variant(base, measurements, method, output, cleanup_stats)
        print(f"{method}: {output.name}; {stats}")


if __name__ == "__main__":
    main()
