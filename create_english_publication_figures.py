from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.metrics import PrecisionRecallDisplay

from model_all_imputations import POLLUTANTS, features
from organize_model_results import split_70_15_15


ROOT=Path(__file__).resolve().parent
OUT=ROOT/"publication_figures_english"; OUT.mkdir(exist_ok=True)
BASE=ROOT/"результаты_моделей"
DATASETS={
    "KNN":ROOT/"Датасет_импутация_KNN.xlsx",
    "linear_interpolation":ROOT/"Датасет_импутация_интерполяцией.xlsx",
    "median":ROOT/"Датасет_импутация_медианой.xlsx",
    "mean":ROOT/"Датасет_импутация_средним.xlsx",
}
MONTH_ORDER=[1,2,3,4,5,6]
MONTH_NAMES=["January","February","March","April","May","June"]


def station_label(x):
    return str(x).replace("ПНЗ №","PNZ ").replace("ПНЗ№","PNZ ").replace("ПНЗ","PNZ").replace("Скат","Skat")


def feature_label(x):
    s=str(x).replace("месяц","Month").replace("день_недели","Day of week").replace("час","Hour").replace("пост_","Station: ")
    return station_label(s)


def heatmap(matrix,title,path,figsize=(10,8),annot=True):
    fig,ax=plt.subplots(figsize=figsize); a=np.asarray(matrix,float)
    im=ax.imshow(a,cmap="coolwarm",aspect="auto",vmin=np.nanmin(a),vmax=np.nanmax(a))
    ax.set_xticks(range(matrix.shape[1]),[station_label(x) for x in matrix.columns],rotation=45,ha="right")
    ax.set_yticks(range(matrix.shape[0]),[station_label(x) for x in matrix.index])
    if annot and matrix.shape[0]<=20 and matrix.shape[1]<=20:
        threshold=np.nanmean(np.abs(a))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(a[i,j]): ax.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(a[i,j])>threshold else "black")
    ax.set_title(title,fontweight="bold"); fig.colorbar(im,ax=ax,shrink=.8); fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)


def importance(names,values,title,path,color="#2878B5"):
    s=pd.Series(values,index=[feature_label(x) for x in names]).sort_values().tail(15)
    fig,ax=plt.subplots(figsize=(10,7)); ax.barh(s.index,s.values,color=color); ax.set_title(title,fontweight="bold"); ax.set_xlabel("Importance"); ax.grid(axis="x",alpha=.2)
    fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)


def shap_importance(model,x,title,path,color):
    sample=x.sample(min(500,len(x)),random_state=42); values=shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values,list): values=values[-1]
    values=np.asarray(values)
    if values.ndim==3: values=values[:,:,-1]
    importance(sample.columns,np.abs(values).mean(axis=0),title,path,color)


def load(path):
    d=pd.read_excel(path,sheet_name="Все наблюдения"); d["Дата_время"]=pd.to_datetime(d["Дата_время"]); return d


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
    data=load(DATASETS["KNN"])
    heatmap(data[POLLUTANTS].corr(method="spearman"),"Spearman Correlations of Core Pollutants",OUT/"01_heatmap_корреляций.png")
    all_ind=[c for c in data.columns[10:] if c!="Количество_импутированных_значений" and pd.api.types.is_numeric_dtype(data[c])]
    heatmap(data[all_ind].corr(method="spearman"),"Spearman Correlations of All 18 Indicators — KNN Imputation",OUT/"01b_heatmap_корреляций_все_18_показателей.png",figsize=(14,12),annot=True)
    station=data.groupby("Пост")[POLLUTANTS].mean().fillna(0); z=(station-station.mean())/station.std().replace(0,1)
    heatmap(z,"Standardized Station–Pollutant Profiles",OUT/"02_heatmap_посты_загрязнители.png",figsize=(11,10))
    fig,axes=plt.subplots(4,2,figsize=(14,16)); axes=axes.ravel()
    for ax,p in zip(axes,POLLUTANTS):
        groups=[data.loc[data["Дата_время"].dt.month.eq(m),p].dropna() for m in MONTH_ORDER]
        ax.boxplot(groups,tick_labels=MONTH_NAMES,showfliers=False); ax.set_title(p,fontweight="bold"); ax.tick_params(axis="x",rotation=30); ax.set_ylabel("Concentration (mg/m³)"); ax.grid(axis="y",alpha=.2)
    axes[-1].axis("off"); fig.suptitle("Monthly Distributions of Core Pollutants",fontsize=15,fontweight="bold"); fig.tight_layout(); fig.savefig(OUT/"03_boxplot_по_месяцам.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    f=data.sort_values(["Пост","Дата_время"]).copy(); f["Target"]=f.groupby("Пост")["NO2"].shift(-1); f=f.dropna(subset=["Target"]).reset_index(drop=True); xf=features(f,POLLUTANTS); _,_,te=split_70_15_15(f)
    fm=joblib.load(BASE/"01_прогноз_NO2"/"KNN_модель.joblib")["model"]; pred=fm.predict(xf.loc[te]); residual=f.loc[te,"Target"].to_numpy()-pred
    importance(xf.columns,fm.feature_importances_,"Feature Importance — Next-Term NO₂ Forecast",OUT/"04_важность_прогноз.png")
    fig,axes=plt.subplots(1,2,figsize=(12,5)); axes[0].scatter(pred,residual,s=9,alpha=.45); axes[0].axhline(0,color="red",ls="--"); axes[0].set(xlabel="Predicted NO₂",ylabel="Residual",title="Residuals vs Predictions"); axes[1].hist(residual,bins=40,color="#2878B5",alpha=.8); axes[1].set(title="Residual Distribution",xlabel="Prediction error"); fig.suptitle("Forecast Error Diagnostics"); fig.tight_layout(); fig.savefig(OUT/"05_остатки_и_ошибки.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    shap_importance(fm,xf.loc[te],"SHAP Importance — Next-Term NO₂ Forecast",OUT/"06_SHAP_прогноз.png","#8172B2")

    cd=data.dropna(subset=["NO2"]).reset_index(drop=True); _,_,te=split_70_15_15(cd); pack=joblib.load(BASE/"04_классификация_высокого_NO2"/"KNN_модель.joblib"); cm=pack["model"]; threshold=pack["threshold"]; xc=features(cd,[p for p in POLLUTANTS if p!="NO2"]); yc=cd["NO2"].gt(threshold).astype(int); prob=cm.predict_proba(xc.loc[te])[:,1]
    importance(xc.columns,cm.feature_importances_,"Feature Importance — High-NO₂ Classification",OUT/"07_важность_классификация.png","#DD8452")
    fig,ax=plt.subplots(figsize=(7,6)); PrecisionRecallDisplay.from_predictions(yc.loc[te],prob,ax=ax); ax.set_title("Precision–Recall Curve — High-NO₂ Classification"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/"08_Precision_Recall.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    shap_importance(cm,xc.loc[te],"SHAP Importance — High-NO₂ Classification",OUT/"09_SHAP_классификация.png","#C44E52")

    sd=data.dropna(subset=["NO2"]).reset_index(drop=True); xs=features(sd,[]); _,_,te=split_70_15_15(sd); sm=joblib.load(BASE/"06_модель_пост_время"/"KNN_модель.joblib")["model"]
    importance(xs.columns,sm.feature_importances_,"Feature Importance — Station–Time NO₂ Model",OUT/"10_важность_пост_время.png","#55A868")
    shap_importance(sm,xs.loc[te],"SHAP Importance — Station–Time NO₂ Model",OUT/"11_SHAP_пост_время.png","#55A868")

    cl=pd.read_excel(BASE/"03_кластеризация_постов"/"KNN_метрики.xlsx",sheet_name="Результаты"); profile=cl.set_index("Пост")[POLLUTANTS]; zz=(profile-profile.mean())/profile.std().replace(0,1); xy=PCA(n_components=2).fit_transform(zz.fillna(0)); labels=cl["Кластер"].to_numpy()
    names={1:"High particulate loading",2:"Low/moderate mixed background",3:"Gas pollution (CO–NOx)",4:"High gas–ozone pollution"}
    fig,ax=plt.subplots(figsize=(10,7));
    for c in sorted(set(labels)):
        m=labels==c; ax.scatter(xy[m,0],xy[m,1],s=90,label=f"Cluster {c}: {names[c]}")
    for i,s in enumerate(cl["Пост"]): ax.annotate(station_label(s),xy[i],xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set(title="Data-Driven Monitoring-Station Clusters",xlabel="PCA 1",ylabel="PCA 2"); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/"12_кластеры_с_названиями.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    for tag,path in DATASETS.items():
        if tag=="KNN": continue
        d=load(path); inds=[c for c in d.columns[10:] if c!="Количество_импутированных_значений" and pd.api.types.is_numeric_dtype(d[c])]
        heatmap(d[inds].corr(method="spearman"),f"Spearman Correlations of All 18 Indicators — {tag.replace('_',' ').title()} Imputation",OUT/f"full18_{tag}.png",figsize=(14,12),annot=True)
    print(OUT)


if __name__=="__main__": main()
