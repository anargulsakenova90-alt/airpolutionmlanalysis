from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from extended_study import features,score,candidates,COMPLETE,add_past_value_control

def main(evidence,reference):
    evidence=Path(evidence);out=evidence/'experiments';reference=Path(reference)
    results=pd.read_csv(out/'rolling_forecast_results.csv')
    city=pd.read_csv(reference/'data/city_primary.csv.gz',parse_dates=['timestamp']).set_index('timestamp')
    bundle=joblib.load(out/'masked_benchmark_ridge_reference.joblib')
    add_past_value_control(city,out,bundle)
    grid=results[(results.horizon==1)&results.algorithm.isin(['Ridge','ExtraTrees','HistGradientBoosting'])].copy()
    grid['tie_method']=grid.method.map({'mean':0,'median':1,'causal_linear':2,'KNN':3})
    grid['tie_features']=grid.feature_set.map({'NO2_only':0,'complete5':1,'without_O3':2,'all7':3})
    selected=grid.sort_values(['validation_RMSE','tie_features','tie_method'],kind='stable').groupby('year',as_index=False).first()
    selected.to_csv(out/'outer_validation_selected.csv',index=False)
    intervals=[]
    for _,w in selected.iterrows():
        d=pd.read_csv(out/f'rolling_{int(w.year)}_{w.feature_set}_{w.method}_{w.algorithm}.csv.gz',parse_dates=['target_time'])
        d['m']=(d.observed-d.predicted)**2;d['p']=(d.observed-d.persistence)**2
        for days in [1,7,28]:
            d['block']=((d.target_time-d.target_time.min()).dt.total_seconds()//(days*86400)).astype(int)
            b=d.groupby('block')[['m','p']].sum().to_numpy();rng=np.random.default_rng(42)
            a=b[rng.integers(0,len(b),size=(2000,len(b)))].sum(axis=1)
            gain=100*(1-np.sqrt(a[:,0]/a[:,1]))
            intervals.append(dict(year=int(w.year),method=w.method,feature_set=w.feature_set,algorithm=w.algorithm,block_days=days,RMSE_gain_pct=100*(1-np.sqrt(d.m.mean()/d.p.mean())),CI95_low=np.quantile(gain,.025),CI95_high=np.quantile(gain,.975),replicates=2000))
    pd.DataFrame(intervals).to_csv(out/'rolling_paired_bootstrap.csv',index=False)
    # Re-evaluate the fixed specified Ridge comparator in both source scenarios.
    sensitivities=[]
    for scenario in ['primary','excluded_zeros','fixed_label_network']:
        frame=pd.read_csv(reference/'data'/f'city_{scenario}.csv.gz',parse_dates=['timestamp']).set_index('timestamp')
        for group,cols in [('complete5',COMPLETE),('NO2_only',['NO2'])]:
            x=features(frame,cols,1);y=frame.NO2.shift(-1);tt=frame.index+pd.Timedelta(hours=1)
            ok=x.notna().all(axis=1)&y.notna();tr=ok&(tt<'2024-07-01');va=ok&(tt>='2024-07-01')&(tt<'2025-01-01');train=ok&(tt<'2025-01-01');test=ok&(tt>='2025-01-01')&(tt<'2026-01-01')
            si=StandardScaler().fit(x.loc[tr]);so=StandardScaler().fit(x.loc[train]);best=None
            for par in candidates('Ridge'):
                model=Ridge(**par).fit(si.transform(x.loc[tr]),y.loc[tr]);err=score(y.loc[va],model.predict(si.transform(x.loc[va])))['RMSE']
                if best is None or err<best[0]:best=(err,par)
            model=Ridge(**best[1]).fit(so.transform(x.loc[train]),y.loc[train]);pred=model.predict(so.transform(x.loc[test]));base=frame.loc[test,'NO2'];s=score(y.loc[test],pred);bs=score(y.loc[test],base)
            sensitivities.append(dict(scenario=scenario,feature_set=group,alpha=best[1]['alpha'],N=s['N'],RMSE=s['RMSE'],R2=s['R2'],baseline_RMSE=bs['RMSE'],gain_pct=100*(1-s['RMSE']/bs['RMSE'])))
    pd.DataFrame(sensitivities).to_csv(out/'ridge_sensitivity.csv',index=False)
    # Error stratification uses training threshold, never a test quantile.
    peak=[]
    for _,w in selected.iterrows():
        d=pd.read_csv(out/f'rolling_{int(w.year)}_{w.feature_set}_{w.method}_{w.algorithm}.csv.gz',parse_dates=['target_time'])
        cutoff=float(city.loc[city.index<pd.Timestamp(f'{int(w.year)}-01-01'),'NO2'].quantile(.9))
        for label,mask in [('training_q90_exceeded',d.observed>cutoff),('below_training_q90',d.observed<=cutoff)]:
            if mask.sum():peak.append(dict(year=int(w.year),stratum=label,threshold=cutoff,model_RMSE=np.sqrt(np.mean((d.loc[mask,'observed']-d.loc[mask,'predicted'])**2)),baseline_RMSE=np.sqrt(np.mean((d.loc[mask,'observed']-d.loc[mask,'persistence'])**2)),N=int(mask.sum())))
    pd.DataFrame(peak).to_csv(out/'peak_error_strata.csv',index=False)
    mask=pd.read_csv(out/'imputation_truth_benchmark.csv');down=pd.read_csv(out/'masked_forecast_benchmark.csv')
    masks=mask.groupby(['method','pollutant','rate','block_hours','mechanism'],as_index=False).agg(mean_NRMSE=('NRMSE_training_sd','mean'),sd_NRMSE=('NRMSE_training_sd','std'),mean_RMSE=('RMSE','mean'),seeds=('seed','nunique'))
    masks.to_csv(out/'imputation_summary.csv',index=False)
    ds=down.groupby(['method','rate','block_hours','mechanism'],as_index=False).agg(mean_RMSE=('RMSE','mean'),sd_RMSE=('RMSE','std'),mean_baseline_RMSE=('baseline_RMSE','mean'),mean_gain_pct=('RMSE_gain_vs_available_persistence_pct','mean'),seeds=('seed','nunique'))
    ds.to_csv(out/'masked_forecast_summary.csv',index=False)
    figures=evidence/'figures';figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,4));ax.errorbar(selected.year,selected.RMSE_gain_vs_persistence_pct,fmt='o',label='Validation-selected model')
    ci=pd.DataFrame(intervals).query('block_days==7').sort_values('year')
    for _,row in ci.iterrows():ax.plot([row.year,row.year],[row.CI95_low,row.CI95_high],color='#2878B5')
    ax.axhline(0,color='red',linestyle='--');ax.set(xlabel='Outer evaluation year',ylabel='RMSE gain over persistence (%)');ax.set_xticks([2022,2023,2024,2025]);ax.legend();fig.tight_layout();fig.savefig(figures/'E1_rolling_skill.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,rate in enumerate([.05,.2]):
        f=masks[(masks.pollutant=='NO2')&(masks.rate==rate)&(masks.mechanism=='independent_channels')]
        for method,g in f.groupby('method'):axes[j].plot(g.block_hours,g.mean_NRMSE,marker='o',label=method)
        axes[j].set(title=f'NO2 recovery, {rate:.0%} masked',xlabel='Requested block length (h)',ylabel='RMSE / training SD');axes[j].set_xticks([1,3,6,24])
    axes[1].legend();fig.tight_layout();fig.savefig(figures/'E2_imputation_truth.png',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,mechanism in enumerate(['independent_channels','synchronized_channels']):
        f=ds[(ds.rate==.1)&(ds.mechanism==mechanism)]
        for method,g in f.groupby('method'):axes[j].plot(g.block_hours,g.mean_RMSE,marker='o',label=method)
        base=f.groupby('block_hours').mean_baseline_RMSE.mean();axes[j].plot(base.index,base,color='black',linestyle='--',label='Available persistence')
        axes[j].set(title=mechanism.replace('_',' '),xlabel='Requested block length (h)',ylabel='Forecast RMSE (source-declared mg/m3)')
    axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'E3_masked_forecasts.png',dpi=300);plt.close(fig)
    horizons=results[results.algorithm.isin(['Ridge_horizon','Persistence','Daily_phase','Weekly_phase'])]
    fig,ax=plt.subplots(figsize=(9,4))
    for algorithm,g in horizons[horizons.feature_set=='complete5'].groupby('algorithm'):
        avg=g.groupby('horizon').RMSE.mean();ax.plot(avg.index,avg,marker='o',label=algorithm)
    ax.set(xlabel='Forecast horizon (h)',ylabel='Mean yearly RMSE (mg/m3)');ax.set_xticks([1,3,6,24]);ax.legend();fig.tight_layout();fig.savefig(figures/'E4_horizons.png',dpi=300);plt.close(fig)
    checks=[dict(check='five masking methods including causal naive control',passed=set(mask.method)=={'mean','median','KNN','causal_linear','past_value'}),dict(check='39 masks x 5 methods x 5 channels',passed=len(mask)==975),dict(check='39 masks x 5 forecast methods',passed=len(down)==195),dict(check='all four outer years',passed=set(selected.year)=={2022,2023,2024,2025}),dict(check='training-only selection recorded',passed=selected.validation_RMSE.notna().all()),dict(check='scenario rows on declared targets',passed=len(sensitivities)==6)]
    for c in checks: c['passed']=bool(c['passed'])
    (out/'EXTENSION_VERIFICATION.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    assert all(c['passed'] for c in checks)
    with pd.ExcelWriter(evidence/'Extended_Results.xlsx') as w:
        for name,file in [('Rolling grid','rolling_forecast_results.csv'),('Inner tuning','inner_tuning_results.csv'),('Selected by inner validation','outer_validation_selected.csv'),('Paired uncertainty','rolling_paired_bootstrap.csv'),('Masked truth','imputation_summary.csv'),('Masked forecasts','masked_forecast_summary.csv'),('Ridge sensitivity','ridge_sensitivity.csv'),('Peak errors','peak_error_strata.csv')]:pd.read_csv(out/file).to_excel(w,sheet_name=name,index=False)
    print('EXTENSION SUMMARIZED',flush=True);print(selected[['year','feature_set','method','algorithm','RMSE','RMSE_gain_vs_persistence_pct']].to_string(index=False));print(pd.DataFrame(sensitivities).to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',required=True);ap.add_argument('--reference',required=True);a=ap.parse_args();main(a.evidence,a.reference)
