from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "результаты_моделей" / "07_полный_доказательный_анализ"


def draw(matrix: pd.DataFrame, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(17, 15))
    image = ax.imshow(matrix.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    ax.set_title(title, fontsize=15, fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=.75, label="Корреляция Спирмена")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    summary = []
    frames = {}
    for path in sorted(ROOT.glob("Датасет_импутация_*.xlsx")):
        tag = path.stem.replace("Датасет_импутация_", "")
        data = pd.read_excel(path, sheet_name="Все наблюдения")
        indicators = list(data.columns[10:])
        correlation = data[indicators].corr(method="spearman")
        frames[tag] = data[indicators]
        output = EVIDENCE / tag / "01b_heatmap_корреляций_все_18_показателей.png"
        draw(correlation, f"Корреляции всех 18 показателей — {tag}", output)
        summary.append({"Импутация": tag, "Показателей": len(indicators), "Файл": str(output)})

    base_name = next(iter(frames))
    base = frames[base_name]
    for name, frame in frames.items():
        summary_row = next(row for row in summary if row["Импутация"] == name)
        summary_row["Ячеек_отличается_от_KNN"] = int(
            (frame.fillna(-999999).sub(base.fillna(-999999)).abs() > 1e-12).sum().sum()
        )
    pd.DataFrame(summary).to_excel(EVIDENCE / "Проверка_различий_heatmap.xlsx", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
