from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "результаты_моделей"
CLUSTER_DIR = MODEL_ROOT / "03_кластеризация_постов"
EVIDENCE_DIR = MODEL_ROOT / "07_полный_доказательный_анализ"
POLLUTANTS = ["PM2.5", "PM10", "SO2", "CO", "NO2", "NO", "O3"]
NAMES = {
    1: "Высокая пылевая нагрузка (PM2.5–PM10)",
    2: "Низкий и умеренный смешанный фон",
    3: "Газовое загрязнение (CO–NOx)",
    4: "Высокое газо-озоновое загрязнение",
}


def rewrite_cluster_report(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    result_name = list(sheets)[1]
    result = sheets[result_name]
    result["Название_кластера"] = result["Кластер"].map(NAMES)
    sheets[result_name] = result
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.book[sheet]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
    return result


def named_plot(data: pd.DataFrame, title: str, output: Path) -> None:
    z = StandardScaler().fit_transform(data[POLLUTANTS].fillna(0))
    xy = PCA(n_components=2).fit_transform(z)
    colors = {1: "#D62728", 2: "#4C78A8", 3: "#F58518", 4: "#54A24B"}
    fig, ax = plt.subplots(figsize=(12, 8))
    for cluster, group in data.groupby("Кластер"):
        ids = group.index.to_numpy()
        ax.scatter(xy[ids, 0], xy[ids, 1], s=90, color=colors[cluster], label=f"{cluster}. {NAMES[cluster]}")
        for i in ids:
            ax.annotate(str(data.loc[i, "Пост"]), xy[i], xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(title=title, xlabel="PCA 1", ylabel="PCA 2")
    ax.grid(alpha=.2); ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def main() -> None:
    mapping = pd.DataFrame({"Кластер": list(NAMES), "Название": list(NAMES.values())})
    mapping.to_excel(CLUSTER_DIR / "названия_кластеров.xlsx", index=False)
    for report in CLUSTER_DIR.glob("*_метрики.xlsx"):
        data = rewrite_cluster_report(report).reset_index(drop=True)
        tag = report.name.replace("_метрики.xlsx", "")
        named_plot(data, f"Именованные кластеры постов — {tag}", CLUSTER_DIR / f"{tag}_кластеры_с_названиями.png")

        evidence = EVIDENCE_DIR / tag
        if evidence.exists():
            named_plot(data, f"Именованные кластеры постов — {tag}", evidence / "12_кластеры_с_названиями.png")
            evidence_reports = list(evidence.glob("*.xlsx"))
            if evidence_reports:
                path = evidence_reports[0]
                sheets = pd.read_excel(path, sheet_name=None)
                cluster_sheet = [s for s in sheets if "Кластер" in s][0]
                frame = sheets[cluster_sheet]
                frame["Название_кластера"] = frame["Кластер"].map(NAMES)
                sheets[cluster_sheet] = frame
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    for sheet, sheet_data in sheets.items():
                        sheet_data.to_excel(writer, sheet_name=sheet, index=False)
    print("Названия кластеров добавлены")


if __name__ == "__main__":
    main()
