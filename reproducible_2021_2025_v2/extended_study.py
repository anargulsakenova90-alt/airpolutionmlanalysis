"""Post-review exploratory robustness experiments; no new untouched test claim."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from corrected_preprocessing import PollutionPreprocessor
from features import CORE

COMPLETE=['PM2.5','PM10','SO2','CO','NO2']
GROUPS={'all7':CORE,'without_O3':[c for c in CORE if c!='O3'],'complete5':COMPLETE,'NO2_only':['NO2']}
METHODS=['mean','median','causal_linear','KNN']

def features(frame,channels,horizon):
    out=pd.DataFrame(index=frame.index)
    for col in channels:
        for lag in ([0,1,3,6,24,48,168] if col=='NO2' else [0,1,3,6]):
            out[f'{col}_lag{lag}']=frame[col].shift(lag)
    for window in [3,6,24,48]:out[f'NO2_mean{window}']=frame.NO2.rolling(window,min_periods=window).mean()
    out['NO2_delta1']=frame.NO2-frame.NO2.shift(1)
    t=frame.index
    out['hour_sin']=np.sin(2*np.pi*t.hour/24);out['hour_cos']=np.cos(2*np.pi*t.hour/24)
    out['doy_sin']=np.sin(2*np.pi*t.dayofyear/365.25);out['doy_cos']=np.cos(2*np.pi*t.dayofyear/365.25)
    out['dow_sin']=np.sin(2*np.pi*t.dayofweek/7);out['dow_cos']=np.cos(2*np.pi*t.dayofweek/7)
    return out

def score(y,p):
    return dict(N=len(y),MAE=float(mean_absolute_error(y,p)),RMSE=float(mean_squared_error(y,p)**.5),R2=float(r2_score(y,p)))
def candidates(name):
    if name=='Ridge':return [dict(alpha=a) for a in [.1,1.,10.,100.]]
    if name=='ExtraTrees':return [dict(min_samples_leaf=l,max_features=f) for l in [5,20] for f in [.8,1.]]
    return [dict(l2_regularization=l,learning_rate=r) for l in [0.,1.] for r in [.05,.1]]
def estimator(name,params):
    if name=='Ridge':return Ridge(**params)
    if name=='ExtraTrees':return ExtraTreesRegressor(n_estimators=60,n_jobs=-1,random_state=42,**params)
    return HistGradientBoostingRegressor(max_iter=100,random_state=42,**params)

def fill(raw,fit_mask,method):
    active=[c for c in CORE if raw.loc[fit_mask,c].notna().any()]
    pre=PollutionPreprocessor(method).fit(raw.loc[fit_mask,active])
    transformed=pre.output_scaler_.inverse_transform(pre.transform(raw[active]))
    filled=pd.DataFrame(transformed,index=raw.index,columns=active)
    # Observed values are preserved exactly, avoiding rounding-only method differences.
    filled=filled.where(raw[active].isna(),raw[active])
    return filled,pre,active

def study(city,out):
    raw=city[CORE];rows=[];tuning=[];predictions=[];cache={};reference=None
    for year in [2022,2023,2024,2025]:
        test_start=pd.Timestamp(f'{year}-01-01');test_end=pd.Timestamp(f'{year+1}-01-01')
        valid_start=test_start-pd.DateOffset(months=6)
        tt=raw.index+pd.Timedelta(hours=1)
        y=raw.NO2.shift(-1)
        fit_inner=tt<valid_start;fit_outer=tt<test_start
        for method in METHODS:
            inner,_,active=fill(raw,fit_inner,method)
            outer,outer_pre,outer_active=fill(raw,fit_outer,method)
            for group,channels in GROUPS.items():
                if group in ['without_O3','NO2_only'] and method!='mean':continue
                used=[c for c in channels if c in active]
                # Drop training-unobserved channels consistently for inner selection
                # and outer refit: new instruments cannot create pretrained effects.
                xin=features(inner,used,1);xout=features(outer,used,1)
                eligible=xin.notna().all(axis=1)&xout.notna().all(axis=1)&y.notna()
                tr=eligible&fit_inner;va=eligible&(tt>=valid_start)&(tt<test_start);train=eligible&fit_outer;test=eligible&(tt>=test_start)&(tt<test_end)
                assert (tt[tr]<valid_start).all() and (tt[va]<test_start).all()
                scaler_in=StandardScaler().fit(xin.loc[tr]);scaler_out=StandardScaler().fit(xout.loc[train])
                a,b=scaler_in.transform(xin.loc[tr]),scaler_in.transform(xin.loc[va])
                atr,ate=scaler_out.transform(xout.loc[train]),scaler_out.transform(xout.loc[test])
                fingerprint=hashlib.sha256(a.tobytes()+b.tobytes()+atr.tobytes()+ate.tobytes()).hexdigest()
                for algorithm in ['Ridge','ExtraTrees','HistGradientBoosting']:
                    key=(year,fingerprint,algorithm)
                    if key in cache:best_params,val_rmse,model,pred=cache[key]
                    else:
                        best=None
                        for params in candidates(algorithm):
                            model=estimator(algorithm,params).fit(a,y.loc[tr]);s=score(y.loc[va],model.predict(b))
                            tuning.append(dict(year=year,method=method,feature_set=group,algorithm=algorithm,parameters=json.dumps(params,sort_keys=True),validation_RMSE=s['RMSE'],training_rows=int(tr.sum()),validation_rows=int(va.sum())))
                            if best is None or s['RMSE']<best[0]:best=(s['RMSE'],params)
                        val_rmse,best_params=best
                        model=estimator(algorithm,best_params).fit(atr,y.loc[train]);pred=model.predict(ate)
                        cache[key]=(best_params,val_rmse,model,pred)
                    s=score(y.loc[test],pred)
                    baseline=raw.loc[test,'NO2']
                    gain=100*(1-s['RMSE']/score(y.loc[test],baseline)['RMSE'])
                    rows.append(dict(year=year,horizon=1,method=method,feature_set=group,algorithm=algorithm,parameters=json.dumps(best_params,sort_keys=True),validation_RMSE=val_rmse,used_channels=','.join(used),features=len(xout.columns),train_N=int(train.sum()),RMSE_gain_vs_persistence_pct=gain,**s))
                    if year==2025 and group=='complete5' and method=='mean' and algorithm=='Ridge':
                        reference=dict(estimator=model,feature_scaler=scaler_out,features=xout.columns.tolist(),parameters=best_params)
                        joblib.dump(reference,out/'masked_benchmark_ridge_reference.joblib',compress=3)
                    details=pd.DataFrame(dict(target_time=tt[test],observed=y.loc[test].to_numpy(),predicted=pred,persistence=baseline.to_numpy()))
                    details.to_csv(out/f'rolling_{year}_{group}_{method}_{algorithm}.csv.gz',index=False,compression='gzip')
                print('Rolling fold',year,group,method,'complete',flush=True)
            pd.DataFrame(rows).to_csv(out/'rolling_forecast_results.csv',index=False)
            pd.DataFrame(tuning).to_csv(out/'inner_tuning_results.csv',index=False)
        # Baselines use only observations available by the issuance time.
        for horizon in [1,3,6,24]:
            target=raw.NO2.shift(-horizon);time_target=raw.index+pd.Timedelta(hours=horizon)
            for group in ['complete5','NO2_only']:
                x=features(raw, GROUPS[group],horizon)
                eligible=x.notna().all(axis=1)&target.notna()
                tr=eligible&(time_target<valid_start);va=eligible&(time_target>=valid_start)&(time_target<test_start);train=eligible&(time_target<test_start);te=eligible&(time_target>=test_start)&(time_target<test_end)
                si=StandardScaler().fit(x.loc[tr]);so=StandardScaler().fit(x.loc[train]);best=None
                for par in candidates('Ridge'):
                    est=Ridge(**par).fit(si.transform(x.loc[tr]),target.loc[tr]);err=score(target.loc[va],est.predict(si.transform(x.loc[va])))['RMSE']
                    if best is None or err<best[0]:best=(err,par)
                est=Ridge(**best[1]).fit(so.transform(x.loc[train]),target.loc[train]);pred=est.predict(so.transform(x.loc[te]))
                baseline=raw.NO2.loc[te];s=score(target.loc[te],pred)
                rows.append(dict(year=year,horizon=horizon,method='none',feature_set=group,algorithm='Ridge_horizon',parameters=json.dumps(best[1]),validation_RMSE=best[0],used_channels=','.join(GROUPS[group]),features=len(x.columns),train_N=int(train.sum()),RMSE_gain_vs_persistence_pct=100*(1-s['RMSE']/score(target.loc[te],baseline)['RMSE']),**s))
                for name,period in [('Persistence',0),('Daily_phase',24),('Weekly_phase',168)]:
                    shift=0 if period==0 else max(period-horizon,0)
                    bp=raw.NO2.shift(shift).loc[te]
                    valid=bp.notna();ss=score(target.loc[te][valid],bp[valid])
                    rows.append(dict(year=year,horizon=horizon,method='none',feature_set=group,algorithm=name,parameters='{}',validation_RMSE=np.nan,used_channels='NO2',features=1,train_N=int(train.sum()),RMSE_gain_vs_persistence_pct=np.nan,**ss))
            print('Horizon',horizon,'year',year,'complete',flush=True)
        pd.DataFrame(rows).to_csv(out/'rolling_forecast_results.csv',index=False)
    return reference

def block_mask(n,rate,length,rng):
    mask=np.zeros(n,dtype=bool);wanted=round(n*rate)
    for start in rng.permutation(n):
        if mask.sum()>=wanted:break
        end=min(n,start+length)
        positions=np.arange(start,end);positions=positions[~mask[positions]]
        remaining=wanted-int(mask.sum());mask[positions[:remaining]]=True
    return mask

def masking_benchmark(city,out,reference):
    raw=city[COMPLETE].copy();train=raw.index<'2025-01-01';test=raw.index>='2025-01-01'
    assert raw.loc[test].notna().all().all()
    methods={m:PollutionPreprocessor(m).fit(raw.loc[train]) for m in METHODS}
    rows=[];downstream=[]
    cases=[(r,l,s,'independent_channels') for r in [.05,.1,.2] for l in [1,3,6,24] for s in [17,29,43]]+[(.1,24,s,'synchronized_channels') for s in [17,29,43]]
    for rate,length,seed,mechanism in cases:
        rng=np.random.default_rng(seed);mask=np.zeros(raw.shape,dtype=bool)
        n=int(test.sum());shared=block_mask(n,rate,length,rng)
        for j in range(len(COMPLETE)):
            mask[test,j]=shared if mechanism=='synchronized_channels' else block_mask(n,rate,length,rng)
        damaged=raw.mask(mask)
        y=raw.NO2.shift(-1);tt=raw.index+pd.Timedelta(hours=1)
        eval_mask=(tt>='2025-01-01')&(tt<'2026-01-01')&y.notna()
        last_available=damaged.NO2.ffill()
        for method,pre in methods.items():
            filled=pd.DataFrame(pre.output_scaler_.inverse_transform(pre.transform(damaged)),index=raw.index,columns=COMPLETE)
            filled=filled.where(damaged.isna(),damaged)
            for j,col in enumerate(COMPLETE):
                mm=mask[:,j];err=filled.loc[mm,col]-raw.loc[mm,col]
                sd=float(raw.loc[train,col].std())
                rows.append(dict(rate=rate,block_hours=length,seed=seed,mechanism=mechanism,method=method,pollutant=col,masked_N=int(mm.sum()),MAE=float(err.abs().mean()),RMSE=float(np.sqrt(np.mean(err**2))),NRMSE_training_sd=float(np.sqrt(np.mean(err**2))/sd)))
            x=features(filled,COMPLETE,1);ok=eval_mask&x.notna().all(axis=1)
            pred=reference['estimator'].predict(reference['feature_scaler'].transform(x.loc[ok,reference['features']]))
            base=last_available.loc[ok];s=score(y.loc[ok],pred);bs=score(y.loc[ok],base)
            downstream.append(dict(rate=rate,block_hours=length,seed=seed,mechanism=mechanism,method=method,baseline_RMSE=bs['RMSE'],RMSE_gain_vs_available_persistence_pct=100*(1-s['RMSE']/bs['RMSE']),**s))
            assert np.allclose(filled.to_numpy()[~mask],raw.to_numpy()[~mask],equal_nan=True)
        print('Masking',mechanism,rate,length,seed,'complete',flush=True)
        pd.DataFrame(rows).to_csv(out/'imputation_truth_benchmark.csv',index=False)
        pd.DataFrame(downstream).to_csv(out/'masked_forecast_benchmark.csv',index=False)

def add_past_value_control(city,out,reference):
    """Additional simple causal imputation control on exactly the same masks."""
    raw=city[COMPLETE].copy();train=raw.index<'2025-01-01';test=raw.index>='2025-01-01'
    rows=[];downstream=[]
    cases=[(r,l,s,'independent_channels') for r in [.05,.1,.2] for l in [1,3,6,24] for s in [17,29,43]]+[(.1,24,s,'synchronized_channels') for s in [17,29,43]]
    for rate,length,seed,mechanism in cases:
        rng=np.random.default_rng(seed);mask=np.zeros(raw.shape,dtype=bool);n=int(test.sum());shared=block_mask(n,rate,length,rng)
        for j in range(len(COMPLETE)):mask[test,j]=shared if mechanism=='synchronized_channels' else block_mask(n,rate,length,rng)
        damaged=raw.mask(mask);filled=damaged.ffill().fillna(raw.loc[train].median())
        for j,col in enumerate(COMPLETE):
            mm=mask[:,j];err=filled.loc[mm,col]-raw.loc[mm,col];sd=float(raw.loc[train,col].std())
            rows.append(dict(rate=rate,block_hours=length,seed=seed,mechanism=mechanism,method='past_value',pollutant=col,masked_N=int(mm.sum()),MAE=float(err.abs().mean()),RMSE=float(np.sqrt(np.mean(err**2))),NRMSE_training_sd=float(np.sqrt(np.mean(err**2))/sd)))
        y=raw.NO2.shift(-1);tt=raw.index+pd.Timedelta(hours=1);x=features(filled,COMPLETE,1)
        ok=(tt>='2025-01-01')&(tt<'2026-01-01')&y.notna()&x.notna().all(axis=1)
        pred=reference['estimator'].predict(reference['feature_scaler'].transform(x.loc[ok,reference['features']]))
        base=damaged.NO2.ffill().loc[ok];s=score(y.loc[ok],pred);bs=score(y.loc[ok],base)
        downstream.append(dict(rate=rate,block_hours=length,seed=seed,mechanism=mechanism,method='past_value',baseline_RMSE=bs['RMSE'],RMSE_gain_vs_available_persistence_pct=100*(1-s['RMSE']/bs['RMSE']),**s))
    for file,data in [('imputation_truth_benchmark.csv',rows),('masked_forecast_benchmark.csv',downstream)]:
        previous=pd.read_csv(out/file);previous=previous[previous.method!='past_value']
        pd.concat([previous,pd.DataFrame(data)],ignore_index=True).to_csv(out/file,index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETED.json').exists():raise RuntimeError('Completed output exists; choose a new directory')
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    city=pd.read_csv(args.city,parse_dates=['timestamp']).set_index('timestamp').sort_index()
    assert city.index.is_unique and (city.index[1:]-city.index[:-1]==pd.Timedelta(hours=1)).all()
    city.groupby(city.index.year)[CORE].agg(lambda x:100*x.isna().mean()).to_csv(out/'missingness_by_year.csv')
    t=time.time();reference=study(city,out);masking_benchmark(city,out,reference);add_past_value_control(city,out,reference)
    result=dict(status='complete',seconds=time.time()-t,outer_years=[2022,2023,2024,2025],interpretation='post-review exploratory analyses on previously inspected data; not new independent confirmation',mask_cases=39,mask_methods=4,truth_targets='observed, never imputed')
    (out/'COMPLETED.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('EXTENDED STUDY COMPLETE',flush=True)

if __name__=='__main__':main()
