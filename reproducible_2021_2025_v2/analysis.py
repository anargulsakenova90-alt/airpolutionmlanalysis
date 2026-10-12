from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor,RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier,IsolationForest
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from corrected_preprocessing import PollutionPreprocessor
import features as old

from paths import CODE as ROOT, OUTPUT as OUT, RAW
OUT.mkdir(parents=True, exist_ok=True)
DATA=OUT/'data'; DATA.mkdir(exist_ok=True)
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
CORE=old.CORE
METHODS=['mean','median','causal_linear','KNN']

def clean_sources():
    panels=[]; summaries=[]; duplicate_records=[]; manual_parts=[]
    for year in range(2021,2026):
        raw=pd.read_csv(RAW/f'raw_observations_{year}.csv.gz',parse_dates=['timestamp'],low_memory=False)
        auto=raw[raw.source_type=='automatic'].copy()
        auto=auto.rename(columns=old.pdk_renames(auto.columns.tolist()))
        for col in CORE:
            if col not in auto: auto[col]=np.nan
        measure=[c for c in auto.columns if c not in ['timestamp','source_type','station_raw','station_key','source_file','folder_year','declared_year','declared_month','actual_year','actual_month','sampling_hour','metadata_mismatch','all_reported_zero','month','year']]
        # Remove repeated vectors before averaging conflicts at identical times.
        exact=auto.duplicated(['station_key','timestamp']+measure)
        dup=auto.duplicated(['station_key','timestamp'],keep=False)
        if dup.any():
            flagged=auto.loc[dup,['station_key','timestamp','source_file']+CORE].copy()
            flagged['exact_repeat_removed']=exact.loc[dup].to_numpy()
            duplicate_records.append(flagged)
        auto=auto.loc[~exact].copy()
        conflicting=int(auto.duplicated(['station_key','timestamp']).sum())
        auto['suspicious_zero']=auto[CORE].notna().sum(axis=1).ge(3)&auto[CORE].fillna(0).eq(0).all(axis=1)
        auto['any_measurement']=auto[CORE].notna().any(axis=1)
        summaries.append(dict(year=year,automatic_imported=len(raw[raw.source_type=='automatic']),exact_repeats_removed=int(exact.sum()),conflicting_extra_records=conflicting,suspicious_zero_records=int(auto.suspicious_zero.sum()),manual_imported=int((raw.source_type=='manual').sum())))
        # Conflict rule: average unique measurement vectors, not repeated source records.
        unique_time=auto.groupby(['station_key','timestamp'],as_index=False)[CORE+['suspicious_zero']].mean()
        for scenario in ['retained_zeros','excluded_zeros']:
            frame=unique_time.copy()
            if scenario=='excluded_zeros':
                # Conservative sensitivity: remove timestamps with any all-core-zero vector.
                frame.loc[frame.suspicious_zero.gt(0),CORE]=np.nan
            frame['timestamp']=frame.timestamp.dt.floor('h')
            h=frame.groupby(['station_key','timestamp'],as_index=False)[CORE].mean()
            h['scenario']=scenario
            panels.append(h)
        manual=raw[raw.source_type=='manual'].copy()
        # Preserve repeated manual blocks under neutral names, never treat them as concentrations.
        rename={c:c[:-2]+'_secondary_block' for c in manual.columns if c.endswith('_2')}
        manual=manual.rename(columns=rename)
        manual=manual.loc[manual[CORE].notna().any(axis=1) if all(c in manual for c in CORE) else manual[[c for c in CORE if c in manual]].notna().any(axis=1)].copy()
        # Suspended matter, phenol and formaldehyde are also concentration measurements.
        manual_original=raw[raw.source_type=='manual'].rename(columns=rename).copy()
        conc=CORE+['Взвешенные вещества','Фенол','Формальдегид','Бензол','Хлорбензол','Этилбензол','Бенз(а)пирен','Параксилол(С2Н6)','Метаксилол(С2Н6)','Ортоксилол(С2Н6)','Кумол(С9Н12)']
        conc=[c for c in conc if c in manual_original]
        valid=manual_original[conc].notna().any(axis=1) if conc else pd.Series(False,index=manual_original.index)
        manual_original['has_primary_concentration']=valid
        manual_original.to_csv(DATA/f'manual_flagged_{year}.csv.gz',index=False,compression='gzip')
        manual_parts.append(manual_original.loc[valid])
        summaries[-1]['manual_primary_concentration_rows']=int(valid.sum())
        print('Source cleanup',year,summaries[-1],flush=True)
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/'source_cleanup.csv',index=False)
    (pd.concat(duplicate_records,ignore_index=True) if duplicate_records else pd.DataFrame(columns=['station_key','timestamp','source_file','exact_repeat_removed']+CORE)).to_csv(OUT/'duplicate_record_audit.csv',index=False)
    if manual_parts: pd.concat(manual_parts,ignore_index=True).to_csv(DATA/'manual_primary_measurements.csv.gz',index=False,compression='gzip')
    panel=pd.concat(panels,ignore_index=True).groupby(['scenario','station_key','timestamp'],as_index=False)[CORE].mean()
    # Reindex each station onto its actual active span: three rows now equal three hours.
    expanded=[]
    for (scenario,station),g in panel.groupby(['scenario','station_key']):
        g=g.set_index('timestamp')[CORE]
        g=g.reindex(pd.date_range(g.index.min(),g.index.max(),freq='h'))
        g.index.name='timestamp'; g=g.reset_index();g['station_key']=station;g['scenario']=scenario;expanded.append(g)
    panel=pd.concat(expanded,ignore_index=True)
    panel.to_csv(DATA/'station_hourly_clean.csv.gz',index=False,compression='gzip')
    return panel,summary

def city_from_panel(panel):
    return old.city_series(panel)

def partition(city):
    lagged,features=old.add_lags(city)
    lagged=lagged[lagged.target_NO2_next_hour.notna()].copy()
    lagged=lagged.set_index('timestamp',drop=False)
    tr=lagged[lagged.target_time<'2024-01-01']
    va=lagged[(lagged.target_time>='2024-01-01')&(lagged.target_time<'2025-01-01')]
    te=lagged[(lagged.target_time>='2025-01-01')&(lagged.target_time<'2026-01-01')]
    return tr,va,te,features

def regressors():
    return dict(Ridge=Ridge(alpha=10),RandomForest=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,max_features=.8,n_jobs=-1,random_state=42),ExtraTrees=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=5,n_jobs=-1,random_state=42),HistGradientBoosting=HistGradientBoostingRegressor(max_iter=150,l2_regularization=1,random_state=42),DecisionTree=DecisionTreeRegressor(max_depth=12,min_samples_leaf=20,random_state=42))
def classifiers():
    return dict(LogisticRegression=LogisticRegression(max_iter=1000,class_weight='balanced'),RandomForest=RandomForestClassifier(n_estimators=100,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=42),ExtraTrees=ExtraTreesClassifier(n_estimators=100,min_samples_leaf=5,class_weight='balanced',n_jobs=-1,random_state=42),HistGradientBoosting=HistGradientBoostingClassifier(max_iter=150,l2_regularization=1,random_state=42),DecisionTree=DecisionTreeClassifier(max_depth=12,min_samples_leaf=20,class_weight='balanced',random_state=42))

def regscore(y,pred):
    return dict(N=len(y),MAE=mean_absolute_error(y,pred),RMSE=mean_squared_error(y,pred)**.5,R2=r2_score(y,pred))
def clscore(y,pred,prob):
    return dict(N=len(y),Accuracy=accuracy_score(y,pred),F1=f1_score(y,pred,zero_division=0),Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred,zero_division=0),ROC_AUC=roc_auc_score(y,prob) if pd.Series(y).nunique()>1 else np.nan,PR_AUC=average_precision_score(y,prob) if pd.Series(y).nunique()>1 else np.nan,prevalence=np.mean(y))

def predict_context(pre,part,features):
    return pre.transform(part[features])

def predictive_grid(city):
    tr,va,te,features=partition(city)
    ytr=tr.target_NO2_next_hour; threshold=float(ytr.quantile(.75))
    rows=[]; selected=[]
    for task in ['forecast','classification']:
        for label,part in [('Validation_2024',va),('Test_2025',te)]:
            persistence=part.NO2.fillna(ytr.median())
            score=regscore(part.target_NO2_next_hour,persistence) if task=='forecast' else clscore((part.target_NO2_next_hour>threshold).astype(int),(persistence>threshold).astype(int),persistence)
            rows.append(dict(task=task,method='none',algorithm='Persistence',subset=label,threshold=threshold,**score))
    for method in METHODS:
        pre=PollutionPreprocessor(method).fit(tr[features])
        xtr=pre.transform(tr[features])
        # A full chronological prefix gives causal extrapolation its past context;
        # fitted statistics/donors/scaling remain strictly training-only.
        allparts=pd.concat([tr,va,te])
        xa=pre.transform(allparts[features]); offset=len(tr)
        xva=xa[offset:offset+len(va)]; xte=xa[offset+len(va):]
        for task,factory in [('forecast',regressors),('classification',classifiers)]:
            for name,est in factory().items():
                train_y=ytr if task=='forecast' else (ytr>threshold).astype(int)
                est.fit(xtr,train_y)
                model=Pipeline([('preprocess',pre),('estimator',est)])
                joblib.dump(dict(pipeline=model,features=features,threshold=threshold,target='observed city NO2 one hour ahead',context='chronological feature rows, include recent history for causal_linear',train_period='2021-2023'),OUT/f'{task}_{method}_{name}.joblib',compress=3)
                for label,part,x in [('Validation_2024',va,xva),('Test_2025',te,xte)]:
                    pred=est.predict(x); prob=est.predict_proba(x)[:,1] if task=='classification' else None
                    target=part.target_NO2_next_hour if task=='forecast' else (part.target_NO2_next_hour>threshold).astype(int)
                    score=regscore(target,pred) if task=='forecast' else clscore(target,pred,prob)
                    rows.append(dict(task=task,method=method,algorithm=name,subset=label,threshold=threshold,**score))
                    details=pd.DataFrame(dict(timestamp=part.timestamp,target_time=part.target_time,observed=target,predicted=pred,persistence=part.NO2))
                    if prob is not None: details['probability']=prob
                    details.to_csv(OUT/f'predictions_{task}_{method}_{name}_{label}.csv.gz',index=False,compression='gzip')
            print('Grid completed',method,task,flush=True)
        pd.DataFrame(rows).to_csv(OUT/'predictive_comparison.csv',index=False)
    result=pd.DataFrame(rows)
    for task,metric in [('forecast','RMSE'),('classification','F1')]:
        candidates=result[(result.task==task)&(result.subset=='Validation_2024')&(result.algorithm!='Persistence')]
        winner=candidates.sort_values(metric,ascending=(task=='forecast')).iloc[0]
        selected.append(winner.to_dict())
    pd.DataFrame(selected).to_csv(OUT/'validation_selected_models.csv',index=False)
    return result,selected,features,threshold

def exploratory_and_station(panel,city):
    rows=[]
    train_city=city[city.timestamp<'2024-01-01']; test_city=city[city.timestamp>='2025-01-01']
    # Identical observed station target rows in every comparison; no target imputation.
    station=panel[panel.NO2.notna()].copy()
    station['hour_sin']=np.sin(2*np.pi*station.timestamp.dt.hour/24);station['hour_cos']=np.cos(2*np.pi*station.timestamp.dt.hour/24)
    station['doy_sin']=np.sin(2*np.pi*station.timestamp.dt.dayofyear/365.25);station['doy_cos']=np.cos(2*np.pi*station.timestamp.dt.dayofyear/365.25)
    tr=station[station.timestamp<'2024-01-01'];te=station[station.timestamp>='2025-01-01']
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(tr[['station_key']])
    cols=['hour_sin','hour_cos','doy_sin','doy_cos']
    xtr=np.column_stack([enc.transform(tr[['station_key']]),tr[cols]])
    xte=np.column_stack([enc.transform(te[['station_key']]),te[cols]])
    model=RandomForestRegressor(n_estimators=80,min_samples_leaf=20,n_jobs=-1,random_state=42).fit(xtr,tr.NO2)
    rows.append(dict(analysis='observed_station_calendar_model',method='no_missing_predictors',subset='Test_2025',**regscore(te.NO2,model.predict(xte))))
    pd.DataFrame(dict(station_key=te.station_key,timestamp=te.timestamp,observed=te.NO2,predicted=model.predict(xte))).to_csv(OUT/'station_observed_target_predictions.csv.gz',index=False,compression='gzip')
    for method in METHODS:
        pre=PollutionPreprocessor(method).fit(train_city[CORE])
        all_x=pre.transform(city[CORE]);trz=all_x[city.timestamp<'2024-01-01'];tez=all_x[city.timestamp>='2025-01-01']
        iso=IsolationForest(n_estimators=150,contamination=.03,n_jobs=-1,random_state=42).fit(trz)
        flags=iso.predict(tez)
        rows.append(dict(analysis='IsolationForest',method=method,subset='Test_2025',N=len(flags),flagged=int((flags==-1).sum()),flagged_pct=float((flags==-1).mean()*100)))
        joblib.dump(dict(pipeline=Pipeline([('preprocess',pre),('estimator',iso)]),features=CORE),OUT/f'anomaly_{method}.joblib',compress=3)
        pca=PCA().fit(trz)
        rows.append(dict(analysis='PCA',method=method,subset='Training_2021_2023',N=len(trz),variance_first3_pct=float(pca.explained_variance_ratio_[:3].sum()*100)))
        if method=='causal_linear':
            print('Exploratory completed',method,'(clustering not applicable)',flush=True)
            continue
        profiles=panel.groupby('station_key')[CORE].mean()
        useful=[c for c in CORE if profiles[c].notna().sum()>=5]
        profiles=profiles[useful];profiles=profiles[profiles.notna().sum(axis=1)>=3]
        prep=PollutionPreprocessor(method if method!='causal_linear' else 'median').fit(profiles)
        z=prep.transform(profiles)
        candidates=[]
        for k in range(2,min(8,len(profiles)-1)+1):
            km=KMeans(n_clusters=k,n_init=20,random_state=42).fit(z);candidates.append((silhouette_score(z,km.labels_),km))
        sil,km=max(candidates,key=lambda a:a[0])
        rows.append(dict(analysis='station_profile_clustering',method=method,subset='Descriptive_all_years',N=len(profiles),silhouette=sil,k=km.n_clusters,profile_imputer=('median' if method=='causal_linear' else method)))
        pd.DataFrame(dict(station_key=profiles.index,cluster=km.labels_)).to_csv(OUT/f'clusters_{method}.csv',index=False)
        print('Exploratory completed',method,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'exploratory_comparison.csv',index=False)

def sensitivity(panel,selected,threshold):
    rows=[]
    base=panel[panel.scenario=='retained_zeros'].copy()
    # Fixed source labels with observed NO2 in every year; station identity is
    # label-based until a physical crosswalk is supplied.
    coverage=base.assign(year=base.timestamp.dt.year).groupby(['station_key','year']).NO2.count().unstack(fill_value=0)
    eligible=coverage.index[coverage.reindex(columns=range(2021,2026),fill_value=0).ge(4380).all(axis=1)]
    pd.DataFrame({'station_key':eligible}).to_csv(OUT/'fixed_network_labels.csv',index=False)
    cases=[('excluded_zeros',panel[panel.scenario=='excluded_zeros'])]
    if len(eligible)>=2:cases.append(('fixed_label_network',base[base.station_key.isin(eligible)]))
    for scenario,frame in cases:
        city=city_from_panel(frame);city.to_csv(DATA/f'city_{scenario}.csv.gz',index=False,compression='gzip')
        tr,va,te,features=partition(city)
        for winner in selected:
            task=winner['task'];method=winner['method'];name=winner['algorithm']
            pre=PollutionPreprocessor(method).fit(tr[features]);xtr=pre.transform(tr[features]);xa=pre.transform(pd.concat([tr,va,te])[features]);xte=xa[len(tr)+len(va):]
            local_threshold=float(tr.target_NO2_next_hour.quantile(.75))
            est=(regressors() if task=='forecast' else classifiers())[name]
            ytr=tr.target_NO2_next_hour if task=='forecast' else (tr.target_NO2_next_hour>local_threshold).astype(int)
            yte=te.target_NO2_next_hour if task=='forecast' else (te.target_NO2_next_hour>local_threshold).astype(int)
            est.fit(xtr,ytr);pred=est.predict(xte)
            score=regscore(yte,pred) if task=='forecast' else clscore(yte,pred,est.predict_proba(xte)[:,1])
            baseline=regscore(yte,te.NO2.fillna(tr.NO2.median())) if task=='forecast' else clscore(yte,(te.NO2.fillna(tr.NO2.median())>local_threshold).astype(int),te.NO2.fillna(tr.NO2.median()))
            rows.append(dict(scenario=scenario,task=task,method=method,algorithm=name,subset='Test_2025',threshold=local_threshold,**score))
            rows.append(dict(scenario=scenario,task=task,method='none',algorithm='Persistence',subset='Test_2025',threshold=local_threshold,**baseline))
        print('Sensitivity completed',scenario,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'sensitivity_comparison.csv',index=False)
    return len(eligible)

def main():
    warnings.filterwarnings('ignore',category=RuntimeWarning)
    panel,summary=clean_sources()
    primary=panel[panel.scenario=='retained_zeros'].copy()
    city=city_from_panel(primary);city.to_csv(DATA/'city_primary.csv.gz',index=False,compression='gzip')
    result,selected,features,threshold=predictive_grid(city)
    exploratory_and_station(primary,city)
    nfixed=sensitivity(panel,selected,threshold)
    with pd.ExcelWriter(OUT/'corrected_comparison.xlsx') as writer:
        for sheet,file in [('Predictive models','predictive_comparison.csv'),('Validation selection','validation_selected_models.csv'),('Sensitivity','sensitivity_comparison.csv'),('Exploratory','exploratory_comparison.csv'),('Source cleanup','source_cleanup.csv')]:
            pd.read_csv(OUT/file).to_excel(writer,sheet_name=sheet,index=False)
    manifest=dict(status='computations_completed',period='2021-2025',train='2021-2023',validation=2024,test=2025,methods=METHODS,forecast_algorithms=list(regressors()),classification_algorithms=list(classifiers()),predictive_combinations=40,knn_training_donor_limit=2000,threshold=threshold,fixed_network_source_labels=nfixed,source_cleanup=summary.to_dict('records'),remaining_limits=['Physical station identities and source measurement units require verification.','All-zero rows are ambiguous; retained and excluded scenarios are both reported.','Manual secondary blocks preserved under neutral labels and excluded from concentration features.','Sensitivity models refit on each scenario; target populations differ.','Causal preprocessing requires chronological context when used on new batches.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('MODEL CALCULATIONS COMPLETE',flush=True)
    import runpy
    runpy.run_path(str(ROOT/'finalize_outputs.py'),run_name='__main__')

if __name__=='__main__':main()
