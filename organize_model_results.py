from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    mean_absolute_error, precision_score, r2_score, recall_score,
    roc_auc_score, root_mean_squared_error, silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from model_all_imputations import POLLUTANTS, RANDOM_STATE, features, safe_name


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "результаты_моделей"
INPUTS = sorted(ROOT.glob("Датасет_импутация_*.xlsx"))
FOLDERS = {
    1: OUT / "01_прогноз_NO2",
    2: OUT / "02_поиск_аномалий",
    3: OUT / "03_кластеризация_постов",
    4: OUT / "04_классификация_высокого_NO2",
    5: OUT / "05_факторный_анализ_PCA",
    6: OUT / "06_модель_пост_время",
}
for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)


def split_70_15_15(data: pd.DataFrame):
    order = data["Дата_время"].sort_values(kind="stable").index.to_numpy()
    a, b = int(len(order) * .70), int(len(order) * .85)
    return order[:a], order[a:b], order[b:]


def reg_metrics(y, pred, subset):
    return {
        "Выборка": subset, "Количество": len(y),
        "MAE": mean_absolute_error(y, pred),
        "RMSE": root_mean_squared_error(y, pred), "R2": r2_score(y, pred),
    }


def save_table(folder: Path, tag: str, metrics: pd.DataFrame, details: pd.DataFrame | None = None, supervised: bool = True):
    path = folder / f"{tag}_метрики.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Метрики", index=False)
        if details is not None:
            details.to_excel(writer, sheet_name="Результаты", index=False)
        if supervised:
            values = ["70%", "15%", "15%", "Хронологически, без перемешивания"]
        else:
            values = ["100%", "—", "—", "Неконтролируемая модель обучена на всех данных"]
        info = pd.DataFrame({
            "Параметр": ["Train/анализ", "Validation", "Test", "Принцип разбиения"],
            "Значение": values,
        })
        info.to_excel(writer, sheet_name="Разбиение", index=False)


def prediction_plot(y, pred, dates, title, output):
    fig, ax = plt.subplots(figsize=(13, 5))
    n = min(400, len(y)); idx = np.arange(n)
    ax.plot(idx, np.asarray(y)[:n], label="Факт", lw=1.6)
    ax.plot(idx, np.asarray(pred)[:n], label="Прогноз", lw=1.4, alpha=.85)
    ax.set(title=title, xlabel="Первые наблюдения тестовой выборки", ylabel="NO₂, мг/м³")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=170); plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    comparison = {i: [] for i in FOLDERS}
    for path in INPUTS:
        data = pd.read_excel(path, sheet_name="Все наблюдения")
        data["Дата_время"] = pd.to_datetime(data["Дата_время"])
        tag = safe_name(path.stem.replace("Датасет_импутация_", ""))

        # 1 — forecast next NO2.
        f = data.sort_values(["Пост", "Дата_время"]).copy()
        f["Цель"] = f.groupby("Пост")["NO2"].shift(-1)
        f = f.dropna(subset=["Цель"]).reset_index(drop=True)
        x, y = features(f, POLLUTANTS), f["Цель"].astype(float)
        tr, va, te = split_70_15_15(f)
        model = RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(x.loc[tr], y.loc[tr]); pv, pt = model.predict(x.loc[va]), model.predict(x.loc[te])
        met = pd.DataFrame([reg_metrics(y.loc[va], pv, "Validation"), reg_metrics(y.loc[te], pt, "Test")])
        det = f.loc[te, ["Дата_время", "Пост", "Цель"]].copy(); det["Прогноз"] = pt
        save_table(FOLDERS[1], tag, met, det); joblib.dump({"model": model, "features": list(x.columns)}, FOLDERS[1] / f"{tag}_модель.joblib")
        prediction_plot(y.loc[te], pt, f.loc[te, "Дата_время"], f"Прогноз NO₂ — {tag}", FOLDERS[1] / f"{tag}_график.png")
        comparison[1].append({"Импутация": tag, **met.iloc[1].to_dict()})

        # 2 — anomalies, 100% data.
        xa = data[POLLUTANTS].fillna(data[POLLUTANTS].median()).fillna(0); scaler = StandardScaler(); z = scaler.fit_transform(xa)
        model = IsolationForest(contamination=.03, random_state=RANDOM_STATE, n_jobs=-1).fit(z)
        score = -model.score_samples(z); flag = model.predict(z).eq(-1) if isinstance(model.predict(z), pd.Series) else model.predict(z) == -1
        met = pd.DataFrame([{"Выборка": "Все данные (100%)", "Количество": len(data), "Аномалий": int(flag.sum()), "Доля_аномалий_%": float(flag.mean()*100), "Средний_score": float(score.mean())}])
        det = data[["Дата_время", "Пост"] + POLLUTANTS].copy(); det["Аномалия"] = flag; det["Score"] = score
        save_table(FOLDERS[2], tag, met, det, supervised=False); joblib.dump({"model": model, "scaler": scaler}, FOLDERS[2] / f"{tag}_модель.joblib")
        fig, ax = plt.subplots(figsize=(13, 5)); ax.scatter(data["Дата_время"], score, c=np.where(flag, "#C44E52", "#2878B5"), s=8, alpha=.65); ax.set(title=f"Аномалии — {tag}", ylabel="Оценка аномальности"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FOLDERS[2] / f"{tag}_график.png", dpi=170); plt.close(fig)
        comparison[2].append({"Импутация": tag, **met.iloc[0].to_dict()})

        # 3 — station clusters, 100% station profiles.
        profile = data.groupby(["Тип_наблюдения", "Пост"])[POLLUTANTS].mean().fillna(0)
        scaler = StandardScaler(); z = scaler.fit_transform(profile); model = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE).fit(z)
        sil = silhouette_score(z, model.labels_); det = profile.reset_index(); det["Кластер"] = model.labels_ + 1
        met = pd.DataFrame([{"Выборка": "Все посты (100%)", "Постов": len(profile), "Кластеров": 4, "Silhouette": sil, "Inertia": model.inertia_}])
        save_table(FOLDERS[3], tag, met, det, supervised=False); joblib.dump({"model": model, "scaler": scaler}, FOLDERS[3] / f"{tag}_модель.joblib")
        xy = PCA(n_components=2).fit_transform(z); fig, ax = plt.subplots(figsize=(9, 7)); ax.scatter(xy[:,0], xy[:,1], c=model.labels_, cmap="tab10", s=80); [ax.annotate(p, xy[i], xytext=(4,4), textcoords="offset points", fontsize=8) for i,p in enumerate(profile.index.get_level_values("Пост"))]; ax.set(title=f"Кластеры постов — {tag}", xlabel="PCA 1", ylabel="PCA 2"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FOLDERS[3] / f"{tag}_график.png", dpi=170); plt.close(fig)
        comparison[3].append({"Импутация": tag, **met.iloc[0].to_dict()})

        # 4 — high NO2 classifier.
        cd = data.dropna(subset=["NO2"]).reset_index(drop=True); tr, va, te = split_70_15_15(cd)
        threshold = float(cd.loc[tr, "NO2"].quantile(.75)); y = cd["NO2"].gt(threshold).astype(int); x = features(cd, [p for p in POLLUTANTS if p != "NO2"])
        model = RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1).fit(x.loc[tr], y.loc[tr])
        rows=[]
        for subset, ids in [("Validation",va),("Test",te)]:
            pred=model.predict(x.loc[ids]); prob=model.predict_proba(x.loc[ids])[:,1]; rows.append({"Выборка":subset,"Количество":len(ids),"Порог_NO2":threshold,"Accuracy":accuracy_score(y.loc[ids],pred),"Precision":precision_score(y.loc[ids],pred,zero_division=0),"Recall":recall_score(y.loc[ids],pred,zero_division=0),"F1":f1_score(y.loc[ids],pred,zero_division=0),"ROC_AUC":roc_auc_score(y.loc[ids],prob)})
        met=pd.DataFrame(rows); pred=model.predict(x.loc[te]); prob=model.predict_proba(x.loc[te])[:,1]; det=cd.loc[te,["Дата_время","Пост","NO2"]].copy(); det["Факт_класс"]=y.loc[te].to_numpy(); det["Прогноз_класс"]=pred; det["Вероятность"]=prob
        save_table(FOLDERS[4],tag,met,det); joblib.dump({"model":model,"features":list(x.columns),"threshold":threshold},FOLDERS[4]/f"{tag}_модель.joblib")
        fig,axes=plt.subplots(1,2,figsize=(12,5)); ConfusionMatrixDisplay.from_predictions(y.loc[te],pred,ax=axes[0],colorbar=False); RocCurveDisplay.from_predictions(y.loc[te],prob,ax=axes[1]); axes[0].set_title("Матрица ошибок"); axes[1].set_title("ROC-кривая"); fig.suptitle(f"Классификация высокого NO₂ — {tag}"); fig.tight_layout(); fig.savefig(FOLDERS[4]/f"{tag}_график.png",dpi=170); plt.close(fig)
        comparison[4].append({"Импутация":tag,**met.iloc[1].to_dict()})

        # 5 — PCA, 100% data.
        xp=data[POLLUTANTS].fillna(data[POLLUTANTS].median()).fillna(0); scaler=StandardScaler(); z=scaler.fit_transform(xp); model=PCA(n_components=len(POLLUTANTS)).fit(z)
        var=pd.DataFrame({"Компонента":np.arange(1,len(POLLUTANTS)+1),"Дисперсия_%":model.explained_variance_ratio_*100,"Накопленная_%":np.cumsum(model.explained_variance_ratio_)*100}); load=pd.DataFrame(model.components_.T,index=POLLUTANTS,columns=[f"PC{i}" for i in range(1,len(POLLUTANTS)+1)]).reset_index(names="Показатель")
        met=pd.DataFrame([{"Выборка":"Все данные (100%)","Количество":len(data),"Дисперсия_PC1_3_%":float(var.loc[:2,"Дисперсия_%"].sum())}]); save_table(FOLDERS[5],tag,met,pd.concat([var,load],axis=1),supervised=False); joblib.dump({"model":model,"scaler":scaler},FOLDERS[5]/f"{tag}_модель.joblib")
        fig,axes=plt.subplots(1,2,figsize=(13,5)); axes[0].bar(var["Компонента"],var["Дисперсия_%"]); axes[0].plot(var["Компонента"],var["Накопленная_%"],marker="o",color="#C44E52"); axes[0].set(title="Дисперсия компонент",xlabel="Компонента",ylabel="%") ; im=axes[1].imshow(model.components_[:3],aspect="auto",cmap="coolwarm"); axes[1].set(yticks=range(3),yticklabels=["PC1","PC2","PC3"],xticks=range(len(POLLUTANTS)),xticklabels=POLLUTANTS,title="Нагрузки первых 3 компонент"); plt.setp(axes[1].get_xticklabels(),rotation=45,ha="right"); fig.colorbar(im,ax=axes[1]); fig.suptitle(f"PCA — {tag}"); fig.tight_layout(); fig.savefig(FOLDERS[5]/f"{tag}_график.png",dpi=170); plt.close(fig)
        comparison[5].append({"Импутация":tag,**met.iloc[0].to_dict()})

        # 6 — station/time NO2 model.
        sd=data.dropna(subset=["NO2"]).reset_index(drop=True); x=features(sd,[]); y=sd["NO2"].astype(float); tr,va,te=split_70_15_15(sd); model=RandomForestRegressor(n_estimators=180,min_samples_leaf=3,random_state=RANDOM_STATE,n_jobs=-1).fit(x.loc[tr],y.loc[tr]); pv,pt=model.predict(x.loc[va]),model.predict(x.loc[te]); met=pd.DataFrame([reg_metrics(y.loc[va],pv,"Validation"),reg_metrics(y.loc[te],pt,"Test")]); det=sd.loc[te,["Дата_время","Пост","NO2"]].copy(); det["Прогноз"]=pt
        save_table(FOLDERS[6],tag,met,det); joblib.dump({"model":model,"features":list(x.columns),"limitation":"Нет координат"},FOLDERS[6]/f"{tag}_модель.joblib"); prediction_plot(y.loc[te],pt,sd.loc[te,"Дата_время"],f"Пост–время → NO₂ — {tag}",FOLDERS[6]/f"{tag}_график.png"); comparison[6].append({"Импутация":tag,**met.iloc[1].to_dict()})

    for i, rows in comparison.items():
        pd.DataFrame(rows).to_excel(FOLDERS[i]/"сравнение_метрик.xlsx",index=False)
    split_info=pd.DataFrame({"Тип модели":["Прогноз NO2","Классификация","Пост-время","Isolation Forest","K-means","PCA"],"Train":["70%","70%","70%","100%","100%","100%"],"Validation":["15%","15%","15%","—","—","—"],"Test":["15%","15%","15%","—","—","—"],"Комментарий":["Хронологически","Хронологически","Хронологически","Неконтролируемая модель","Неконтролируемая модель","Неконтролируемая модель"]})
    split_info.to_excel(OUT/"пропорции_train_validation_test.xlsx",index=False)
    print(f"Готово: {OUT}")


if __name__ == "__main__":
    main()
