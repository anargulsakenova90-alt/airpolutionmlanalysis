"""Generate all current main and supplementary figures/tables from saved results."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,average_precision_score
from features import CORE,add_lags

def main(results,assets):
    results=Path(results);assets=Path(assets);assets.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(results/'predictive_comparison.csv')
    selected=pd.read_csv(results/'validation_selected_models.csv')
    city=pd.read_csv(results/'data'/'city_primary.csv.gz',parse_dates=['timestamp'])
    panel=pd.read_csv(results/'data'/'station_hourly_clean.csv.gz',parse_dates=['timestamp'])
    panel=panel[panel.scenario=='retained_zeros'].copy()
    figures=[]
    def save(fig,name,caption):
        fig.tight_layout();fig.savefig(assets/f'{name}.png',dpi=300,bbox_inches='tight');plt.close(fig)
        figures.append(dict(id=name,file=f'{name}.png',caption=caption))
    def bar(values,names,name,caption,xlabel):
        idx=np.argsort(values)[-12:]
        fig,ax=plt.subplots(figsize=(8,4.8));ax.barh(np.asarray(names)[idx],np.asarray(values)[idx],color='#2878B5');ax.set_xlabel(xlabel)
        save(fig,name,caption)
    def correlation(method,name):
        f=pd.read_csv(results/'data'/f'features_imputed_{method}.csv.gz')
        columns=[f'{c}_lag0' for c in CORE]
        c=f[columns].corr(method='spearman')
        fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(c,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(7),CORE,rotation=30);ax.set_yticks(range(7),CORE)
        for i in range(7):
            for j in range(7):ax.text(j,i,f'{c.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=ax,label='Spearman correlation')
        save(fig,name,f'Spearman correlations of the seven city-level pollutant features after {method} imputation. Associations are descriptive; the full-period display does not enter predictive fitting.')
    correlation('KNN','S1')
    profiles=panel.groupby('station_key')[CORE].mean()
    z=(profiles-profiles.mean())/profiles.std()
    fig,ax=plt.subplots(figsize=(8,11));im=ax.imshow(z,cmap='coolwarm',aspect='auto',vmin=-3,vmax=3)
    ax.set_xticks(range(7),CORE);ax.set_yticks(range(len(z)),z.index.str.replace('automatic|','',regex=False),fontsize=7)
    fig.colorbar(im,ax=ax,label='Standardized observed station mean')
    save(fig,'S2','Observed station-by-pollutant profiles. Missing cells are not recovered for this display; different labels are not assumed to represent distinct physical locations.')
    fig,ax=plt.subplots(figsize=(8,4.5));ax.boxplot([city.loc[city.timestamp.dt.month==m,'NO2'].dropna() for m in range(1,13)],tick_labels=range(1,13),showfliers=False)
    ax.set(xlabel='Month (all five years pooled)',ylabel='NO2 (source units)')
    save(fig,'S3','Pooled monthly NO2 distributions. Variation reflects seasonality and potentially changing station coverage; it is not a causal seasonal effect.')
    lagged,features=add_lags(city);lagged=lagged[lagged.target_NO2_next_hour.notna()].set_index('timestamp',drop=False)
    test=lagged[lagged.target_time>='2025-01-01']
    bundles={}
    for task in ['forecast','classification']:
        winner=selected[selected.task==task].iloc[0]
        bundle=joblib.load(results/f'{task}_{winner.method}_{winner.algorithm}.joblib');bundles[task]=bundle
        pred=pd.read_csv(results/f'predictions_{task}_{winner.method}_{winner.algorithm}_Test_2025.csv.gz',parse_dates=['target_time'])
        est=bundle['pipeline'].named_steps['estimator']
        x=bundle['pipeline'].named_steps['preprocess'].transform(lagged[bundle['features']])[-len(test):]
        if task=='forecast':
            bar(est.feature_importances_,features,'S4','Impurity-based importance for the validation-selected Extra Trees forecast. Correlated lagged predictors can divide or bias importance.','Impurity importance')
            residual=pred.observed-pred.predicted
            fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].scatter(pred.predicted,residual,s=3,alpha=.25);axes[0].axhline(0,color='red');axes[0].set(xlabel='Prediction',ylabel='Observed minus predicted');axes[1].hist(residual,bins=60);axes[1].set(xlabel='Residual',ylabel='Hours')
            save(fig,'S5','Test residuals and residual distribution for the validation-selected forecast; spread and tails diagnose prediction error without proving calibrated uncertainty.')
        else:
            importances=[]
            # Histogram boosting lacks impurity importance; a held-out permutation
            # score drop is reported, never substituted with another estimator.
            from sklearn.inspection import permutation_importance
            sample=np.arange(0,len(test),max(1,len(test)//1000))
            perm=permutation_importance(est,x[sample],pred.observed.iloc[sample],scoring='f1',n_repeats=3,random_state=42,n_jobs=1)
            bar(perm.importances_mean,features,'S7','Permutation F1 importance for the selected histogram-gradient-boosting classifier on a deterministic test subsample (three repeats). This post-test explanation does not select the model.','Mean F1 decrease after permutation')
            fig,ax=plt.subplots(figsize=(7,4.5))
            for label,score in [('Selected classifier',pred.probability),('Persistence ranking',pred.persistence)]:
                precision,recall,_=precision_recall_curve(pred.observed,score);ap=average_precision_score(pred.observed,score);ax.plot(recall,precision,label=f'{label}, AP={ap:.3f}')
            ax.axhline(pred.observed.mean(),color='gray',linestyle='--',label='Test prevalence');ax.set(xlabel='Recall',ylabel='Precision');ax.legend()
            save(fig,'S8','Precision-recall curves for the selected classifier and current-NO2 persistence ranking. The horizontal reference is test positive prevalence; thresholds are training-defined.')
        import shap
        sample=np.linspace(0,len(x)-1,100,dtype=int)
        explainer=shap.TreeExplainer(est)
        values=explainer.shap_values(x[sample])
        values=np.asarray(values)
        if values.ndim==3:values=values[:,:,1]
        importance=np.mean(np.abs(values),axis=0)
        fig_id='S6' if task=='forecast' else 'S9'
        output='concentration predictions' if task=='forecast' else 'raw classifier margin (log odds)'
        bar(importance,features,fig_id,f'Mean absolute TreeSHAP attribution for {output}, using 100 evenly spaced 2025 test rows. Attribution describes this fitted model and is not causal evidence.','Mean absolute SHAP value')
    station=pd.read_csv(results/'station_observed_target_predictions.csv.gz',parse_dates=['timestamp'])
    station['abs_error']=(station.observed-station.predicted).abs()
    error=station.groupby('station_key').abs_error.mean().sort_values()
    fig,ax=plt.subplots(figsize=(8,6));ax.barh(error.index.str.replace('automatic|','',regex=False),error, color='#C44E52');ax.set_xlabel('Observed-target station MAE')
    save(fig,'S10','Station-level test MAE for the calendar-and-station diagnostic model, using observed NO2 targets only. This replaces the superseded six-month importance figure.')
    grouped=station.groupby('station_key')
    stats=grouped.agg(observed_mean=('observed','mean'),predicted_mean=('predicted','mean'),N=('observed','size'))
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(stats.observed_mean,stats.predicted_mean,s=30);maxval=float(stats[['observed_mean','predicted_mean']].max().max());ax.plot([0,maxval],[0,maxval],'r--');ax.set(xlabel='Observed station test mean',ylabel='Predicted station test mean')
    save(fig,'S11','Observed versus predicted station-average NO2 on the test set. Departures from the diagonal show station-level bias; this is an error diagnostic, not SHAP.')
    clusters=pd.read_csv(results/'clusters_median.csv').set_index('station_key')
    joined=profiles.join(clusters).dropna(subset=['cluster']);centroids=joined.groupby('cluster')[CORE].mean()
    centroid_z=(centroids-profiles.mean())/profiles.std()
    fig,ax=plt.subplots(figsize=(8,4.5));im=ax.imshow(centroid_z,cmap='coolwarm',aspect='auto',vmin=-3,vmax=3);ax.set_xticks(range(7),CORE);ax.set_yticks(range(len(centroid_z)),[f'Cluster {int(k)+1}' for k in centroid_z.index]);fig.colorbar(im,ax=ax,label='Standardized observed centroid')
    save(fig,'S12','Median-imputation station-profile cluster centroids displayed in observed pollutant units after standardization. Cluster numbers are analytical groups, not inferred emission-source names.')
    correlation('causal_linear','S13');correlation('median','S14');correlation('mean','S15')
    # Main figures are reproducible metric/trend displays.
    for task,metric,title in [('forecast','RMSE','Next-hour-bin forecast'),('classification','F1','Relative high-NO2 classification')]:
        tab=metrics[(metrics.task==task)&(metrics.subset=='Test_2025')&(metrics.algorithm!='Persistence')].pivot(index='method',columns='algorithm',values=metric)
        fig,ax=plt.subplots(figsize=(10,4));im=ax.imshow(tab,cmap='viridis_r' if task=='forecast' else 'viridis',aspect='auto');ax.set_xticks(range(len(tab.columns)),tab.columns,rotation=15,ha='right');ax.set_yticks(range(len(tab.index)),tab.index)
        for i in range(len(tab.index)):
            for j in range(len(tab.columns)):ax.text(j,i,f'{tab.iloc[i,j]:.5f}',ha='center',va='center',fontsize=8,color='white')
        ax.set_title(title+' (2025)');fig.colorbar(im,ax=ax,label=metric)
        save(fig,'M1' if task=='forecast' else 'M2',f'All twenty {task} combinations on 2025 test data. The display is descriptive; selection uses 2024 validation.')
    fig,ax=plt.subplots(figsize=(11,4))
    for scenario in ['primary','excluded_zeros']:
        f=pd.read_csv(results/'data'/f'city_{scenario}.csv.gz',parse_dates=['timestamp']).set_index('timestamp');ax.plot(f.NO2.resample('MS').mean(),label=scenario)
    ax.set(xlabel='Date',ylabel='NO2 (source units)');ax.legend()
    save(fig,'M3','Monthly city NO2 with ambiguous all-core-zero records retained and excluded. Differences reflect a source-quality sensitivity, not correction of confirmed outages.')
    tables={
        'S1':pd.read_csv(results/'source_cleanup.csv'),
        'S2':pd.DataFrame([dict(method=m,rows=len(pd.read_csv(results/'data'/f'features_imputed_{m}.csv.gz')),missing_feature_cells=int(pd.read_csv(results/'data'/f'features_imputed_{m}.csv.gz')[features].isna().sum().sum())) for m in ['mean','median','causal_linear','KNN']]),
        'S3':metrics[metrics.task=='forecast'],
        'S4':metrics[metrics.task=='classification'],
        'S5':pd.read_csv(results/'exploratory_comparison.csv'),
        'S6':pd.read_csv(results/'selected_forecast_bootstrap.csv'),
        'S7':pd.read_csv(results/'clusters_median.csv'),
        'S8':pd.read_csv(results/'verification_checks.csv'),
    }
    for name,frame in tables.items():frame.to_csv(assets/f'Table_{name}.csv',index=False)
    pd.DataFrame(figures).to_csv(assets/'figure_manifest.csv',index=False)
    (assets/'asset_provenance.json').write_text(json.dumps(dict(results=str(results),shap_rows=100,permutation_repeats=3,permutation_rows_approx=1000,figures=len(figures),supplementary_tables=8),indent=2),encoding='utf-8')
    print('Publication assets generated:',len(figures),'figures and',len(tables),'supplementary tables',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--assets',required=True)
    args=ap.parse_args();main(args.results,args.assets)
