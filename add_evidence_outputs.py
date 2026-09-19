from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import PrecisionRecallDisplay

from model_all_imputations import POLLUTANTS, features, safe_name
from organize_model_results import split_70_15_15

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
BASE = ROOT / "результаты_моделей"
EVIDENCE = BASE / "07_полный_доказательный_анализ"
EVIDENCE.mkdir(parents=True, exist_ok=True)
INPUTS = sorted(ROOT.glob("Датасет_импутация_*.xlsx"))
MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь"]


def heatmap(matrix, title, output, fmt=".2f", figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    arr = np.asarray(matrix, dtype=float)
    im = ax.imshow(arr, cmap="coolwarm", aspect="auto", vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    if matrix.shape[0] <= 20 and matrix.shape[1] <= 20:
        threshold = np.nanmean(arr)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center", fontsize=7,
                            color="white" if abs(arr[i, j]) > abs(threshold) else "black")
    ax.set_title(title, fontweight="bold"); fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def importance_plot(names, values, title, output, color="#2878B5"):
    s = pd.Series(values, index=names).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(s.index, s.values, color=color)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Важность"); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def shap_values(model, x):
    sample = x.sample(min(500, len(x)), random_state=42)
    values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values, list): values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3: values = values[:, :, -1]
    return sample, np.abs(values).mean(axis=0)


def confidence_intervals(data):
    rows=[]
    for pollutant in POLLUTANTS:
        for month, g in data.groupby("Месяц_файла"):
            x=g[pollutant].dropna().to_numpy(float); n=len(x); mean=float(x.mean())
            margin=float(stats.t.ppf(.975,n-1)*stats.sem(x)) if n>1 else np.nan
            rows.append({"Показатель":pollutant,"Месяц":month,"N":n,"Среднее":mean,"CI95_нижняя":mean-margin,"CI95_верхняя":mean+margin})
    return pd.DataFrame(rows)


def pvalue_tables(data):
    month_rows=[]; station_rows=[]
    for p in POLLUTANTS:
        mg=[g[p].dropna().values for _,g in data.groupby("Месяц_файла") if g[p].notna().sum()>1]
        sg=[g[p].dropna().values for _,g in data.groupby("Пост") if g[p].notna().sum()>1]
        mh,mp=stats.kruskal(*mg) if len(mg)>1 else (np.nan,np.nan)
        sh,sp=stats.kruskal(*sg) if len(sg)>1 else (np.nan,np.nan)
        month_rows.append({"Показатель":p,"Тест":"Kruskal-Wallis по месяцам","Статистика_H":mh,"p_value":mp,"Значимо_p<0.05":mp<.05})
        station_rows.append({"Показатель":p,"Тест":"Kruskal-Wallis по постам","Статистика_H":sh,"p_value":sp,"Значимо_p<0.05":sp<.05})
    corr=[]
    for i,a in enumerate(POLLUTANTS):
        for b in POLLUTANTS[i+1:]:
            pair=data[[a,b]].dropna(); r,pv=stats.spearmanr(pair[a],pair[b]) if len(pair)>2 else (np.nan,np.nan)
            corr.append({"Показатель_1":a,"Показатель_2":b,"Spearman_r":r,"p_value":pv,"N":len(pair),"Значимо_p<0.05":pv<.05})
    return pd.DataFrame(month_rows),pd.DataFrame(station_rows),pd.DataFrame(corr)


def regression_evidence(data):
    d=data.dropna(subset=["NO2"]).copy()
    predictors=[p for p in POLLUTANTS if p!="NO2"]
    x=d[predictors].copy()
    x["месяц"]=d["Дата_время"].dt.month; x["час"]=d["Дата_время"].dt.hour
    x=x.fillna(x.median()); x=sm.add_constant(x.astype(float)); y=d["NO2"].astype(float)
    ols=sm.OLS(y,x).fit(cov_type="HC3")
    ci=ols.conf_int(); table=pd.DataFrame({"Признак":ols.params.index,"Коэффициент":ols.params.values,"p_value":ols.pvalues.values,"CI95_нижняя":ci[0].values,"CI95_верхняя":ci[1].values,"Значимо_p<0.05":ols.pvalues.values<.05})
    global_stats=pd.DataFrame([{"R2":ols.rsquared,"R2_adj":ols.rsquared_adj,"F_p_value":ols.f_pvalue,"N":int(ols.nobs)}])
    return table,global_stats


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
    index_rows=[]
    for path in INPUTS:
        data=pd.read_excel(path,sheet_name="Все наблюдения"); data["Дата_время"]=pd.to_datetime(data["Дата_время"])
        tag=safe_name(path.stem.replace("Датасет_импутация_","")); out=EVIDENCE/tag; out.mkdir(exist_ok=True)

        corr=data[POLLUTANTS].corr(method="spearman")
        heatmap(corr,f"Корреляции Спирмена — {tag}",out/"01_heatmap_корреляций.png")
        station=data.groupby("Пост")[POLLUTANTS].mean().fillna(0)
        standardized=(station-station.mean())/station.std().replace(0,1)
        heatmap(standardized,f"Профили постов (z-score) — {tag}",out/"02_heatmap_посты_загрязнители.png",figsize=(11,10))

        fig,axes=plt.subplots(4,2,figsize=(14,16)); axes=axes.ravel()
        for ax,p in zip(axes,POLLUTANTS):
            groups=[data.loc[data["Месяц_файла"].str.lower().eq(m),p].dropna() for m in MONTHS]
            ax.boxplot(groups,tick_labels=MONTHS,showfliers=False); ax.set_title(p,fontweight="bold"); ax.tick_params(axis="x",rotation=35); ax.grid(axis="y",alpha=.2)
        axes[-1].axis("off"); fig.suptitle(f"Распределения по месяцам — {tag}",fontsize=15,fontweight="bold"); fig.tight_layout(); fig.savefig(out/"03_boxplot_по_месяцам.png",dpi=180); plt.close(fig)

        # Forecast evidence.
        f=data.sort_values(["Пост","Дата_время"]).copy(); f["Цель"]=f.groupby("Пост")["NO2"].shift(-1); f=f.dropna(subset=["Цель"]).reset_index(drop=True); x=features(f,POLLUTANTS); _,_,te=split_70_15_15(f)
        pack=joblib.load(BASE/"01_прогноз_NO2"/f"{tag}_модель.joblib"); model=pack["model"]; pred=model.predict(x.loc[te]); residual=f.loc[te,"Цель"].to_numpy()-pred
        importance_plot(x.columns,model.feature_importances_,f"Важность признаков прогноза — {tag}",out/"04_важность_прогноз.png")
        fig,axes=plt.subplots(1,2,figsize=(12,5)); axes[0].scatter(pred,residual,s=9,alpha=.45); axes[0].axhline(0,color="red",ls="--"); axes[0].set(xlabel="Прогноз",ylabel="Остаток",title="Остатки vs прогноз"); axes[1].hist(residual,bins=40,color="#2878B5",alpha=.8); axes[1].set(title="Распределение ошибок",xlabel="Ошибка"); fig.suptitle(f"Диагностика прогноза — {tag}"); fig.tight_layout(); fig.savefig(out/"05_остатки_и_ошибки.png",dpi=180); plt.close(fig)
        sample,sv=shap_values(model,x.loc[te]); importance_plot(sample.columns,sv,f"SHAP прогноза — {tag}",out/"06_SHAP_прогноз.png",color="#8172B2")

        # Classification evidence and PR curve.
        cd=data.dropna(subset=["NO2"]).reset_index(drop=True); tr,_,te=split_70_15_15(cd); pack=joblib.load(BASE/"04_классификация_высокого_NO2"/f"{tag}_модель.joblib"); model=pack["model"]; threshold=pack["threshold"]; xc=features(cd,[p for p in POLLUTANTS if p!="NO2"]); yc=cd["NO2"].gt(threshold).astype(int); prob=model.predict_proba(xc.loc[te])[:,1]
        importance_plot(xc.columns,model.feature_importances_,f"Важность классификации — {tag}",out/"07_важность_классификация.png",color="#DD8452")
        fig,ax=plt.subplots(figsize=(7,6)); PrecisionRecallDisplay.from_predictions(yc.loc[te],prob,ax=ax); ax.set_title(f"Precision–Recall — {tag}"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/"08_Precision_Recall.png",dpi=180); plt.close(fig)
        sample,sv=shap_values(model,xc.loc[te]); importance_plot(sample.columns,sv,f"SHAP классификации — {tag}",out/"09_SHAP_классификация.png",color="#C44E52")

        # Station-time evidence.
        sd=data.dropna(subset=["NO2"]).reset_index(drop=True); xs=features(sd,[]); _,_,te=split_70_15_15(sd); pack=joblib.load(BASE/"06_модель_пост_время"/f"{tag}_модель.joblib"); model=pack["model"]
        importance_plot(xs.columns,model.feature_importances_,f"Важность пост–время — {tag}",out/"10_важность_пост_время.png",color="#55A868")
        sample,sv=shap_values(model,xs.loc[te]); importance_plot(sample.columns,sv,f"SHAP пост–время — {tag}",out/"11_SHAP_пост_время.png",color="#55A868")

        # Cluster interpretation from existing results.
        cluster_file=BASE/"03_кластеризация_постов"/f"{tag}_метрики.xlsx"; clusters=pd.read_excel(cluster_file,sheet_name="Результаты")
        interpretations=[]
        for cl,g in clusters.groupby("Кластер"):
            means=g[POLLUTANTS].mean().sort_values(ascending=False); interpretations.append({"Кластер":cl,"Количество_постов":len(g),"Посты":", ".join(g["Пост"].astype(str)),"Доминирующие_показатели":", ".join(means.head(3).index),"Максимальный_средний_показатель":means.index[0],"Значение":means.iloc[0]})
        interpretations=pd.DataFrame(interpretations)

        month_p,station_p,corr_p=pvalue_tables(data); ci=confidence_intervals(data); ols,ols_global=regression_evidence(data)
        narrative=pd.DataFrame({"Раздел":["Корреляции","Различия по месяцам","Различия по постам","Регрессия OLS","Доверительные интервалы","SHAP","Ограничение"],"Интерпретация":[f"Максимальная абсолютная корреляция вне диагонали: {corr.where(~np.eye(len(corr),dtype=bool)).abs().stack().idxmax()} = {corr.where(~np.eye(len(corr),dtype=bool)).abs().stack().max():.3f}",f"Значимых показателей: {int(month_p['Значимо_p<0.05'].sum())} из {len(month_p)}",f"Значимых показателей: {int(station_p['Значимо_p<0.05'].sum())} из {len(station_p)}",f"Глобальный F-test p-value={ols_global.at[0,'F_p_value']:.3e}; R²={ols_global.at[0,'R2']:.3f}","95% CI рассчитаны для средних по каждому месяцу","Средняя абсолютная величина SHAP показывает вклад признака, но не причинность","Множественные тесты требуют поправки; p-value не доказывает причинность"]})

        report=out/f"Доказательный_отчёт_{tag}.xlsx"
        with pd.ExcelWriter(report,engine="openpyxl") as w:
            narrative.to_excel(w,sheet_name="Интерпретация",index=False); month_p.to_excel(w,sheet_name="p_month",index=False); station_p.to_excel(w,sheet_name="p_station",index=False); corr_p.to_excel(w,sheet_name="p_correlation",index=False); ols.to_excel(w,sheet_name="OLS_p_values",index=False); ols_global.to_excel(w,sheet_name="OLS_общая",index=False); ci.to_excel(w,sheet_name="CI95",index=False); interpretations.to_excel(w,sheet_name="Кластеры_интерпретация",index=False)
        index_rows.append({"Импутация":tag,"Графиков":len(list(out.glob('*.png'))),"Отчёт":str(report),"Значимых_месячных_тестов":int(month_p['Значимо_p<0.05'].sum()),"Значимых_постовых_тестов":int(station_p['Значимо_p<0.05'].sum()),"OLS_F_p_value":ols_global.at[0,'F_p_value']})
        print(tag,"готово")
    pd.DataFrame(index_rows).to_excel(EVIDENCE/"Сводка_доказательных_материалов.xlsx",index=False)
    print("Готово:",EVIDENCE)


if __name__=="__main__": main()
