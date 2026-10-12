from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches
from corrected_preprocessing import PollutionPreprocessor
import analysis as run

from paths import CODE as ROOT, OUTPUT as OUT
FIG=OUT/'figures'
manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
results=pd.read_csv(OUT/'predictive_comparison.csv')
selected=pd.read_csv(OUT/'validation_selected_models.csv')
sensitivity=pd.read_csv(OUT/'sensitivity_comparison.csv')
cleanup=pd.read_csv(OUT/'source_cleanup.csv')
exploratory=pd.read_csv(OUT/'exploratory_comparison.csv')
exploratory=exploratory[~((exploratory.analysis=='station_profile_clustering')&(exploratory.method=='causal_linear'))]
exploratory.to_csv(OUT/'exploratory_comparison.csv',index=False)
causal_cluster=OUT/'clusters_causal_linear.csv'
if causal_cluster.exists():causal_cluster.unlink()
checks=[]
def check(name,ok,detail):
    checks.append(dict(check=name,status='PASS' if ok else 'FAIL',details=str(detail)))
check('40 predictive combinations',len(results[results.algorithm!='Persistence'][['task','method','algorithm']].drop_duplicates())==40,'20 forecasts plus 20 classifiers')
panel=pd.read_csv(OUT/'data'/'station_hourly_clean.csv.gz',parse_dates=['timestamp'])
check('unique scenario station hours',not panel.duplicated(['scenario','station_key','timestamp']).any(),len(panel))
check('hourly station grids',all(g.timestamp.sort_values().diff().dropna().eq(pd.Timedelta(hours=1)).all() for _,g in panel.groupby(['scenario','station_key'])),'continuous within active label span')
city=pd.read_csv(OUT/'data'/'city_primary.csv.gz',parse_dates=['timestamp'])
tr,va,te,features=run.partition(city)
check('training target boundary',tr.target_time.max()<pd.Timestamp('2024-01-01'),str(tr.target_time.max()))
check('validation target boundary',va.target_time.min()>=pd.Timestamp('2024-01-01') and va.target_time.max()<pd.Timestamp('2025-01-01'),'2024')
check('test target boundary',te.target_time.min()>=pd.Timestamp('2025-01-01') and te.target_time.max()<pd.Timestamp('2026-01-01'),'2025')
check('manual primary filter',len(pd.read_csv(OUT/'data'/'manual_primary_measurements.csv.gz',low_memory=False))==int(cleanup.manual_primary_concentration_rows.sum()),int(cleanup.manual_primary_concentration_rows.sum()))
allparts=pd.concat([tr,va,te])
manual=pd.read_csv(OUT/'data'/'manual_primary_measurements.csv.gz',low_memory=False)
check('manual primary unique station-time',not manual.duplicated(['station_key','timestamp']).any(),len(manual))
for method in run.METHODS:
    bundle=joblib.load(OUT/f'forecast_{method}_Ridge.joblib')
    pre=bundle['pipeline'].named_steps['preprocess']
    filled=pre.output_scaler_.inverse_transform(pre.transform(allparts[features]))
    frame=pd.DataFrame(filled,columns=features,index=allparts.index)
    frame['timestamp']=allparts.timestamp;frame['target_time']=allparts.target_time
    frame['observed_target_NO2']=allparts.target_NO2_next_hour
    frame['split']=np.where(allparts.target_time<'2024-01-01','train',np.where(allparts.target_time<'2025-01-01','validation','test'))
    frame.to_csv(OUT/'data'/f'features_imputed_{method}.csv.gz',index=False,compression='gzip')
    observed=allparts[features].notna().to_numpy()
    check(f'{method} imputed features preserve observations',np.allclose(filled[observed],allparts[features].to_numpy()[observed]),int((~observed).sum()))
for _,row in results[(results.subset=='Test_2025')&(results.algorithm!='Persistence')].iterrows():
    saved=pd.read_csv(OUT/f'predictions_{row.task}_{row.method}_{row.algorithm}_Test_2025.csv.gz',parse_dates=['timestamp','target_time'])
    score=run.regscore(saved.observed,saved.predicted) if row.task=='forecast' else run.clscore(saved.observed,saved.predicted,saved.probability)
    key='RMSE' if row.task=='forecast' else 'F1'
    check(f'{row.task} {row.method} {row.algorithm} metrics',np.isclose(score[key],row[key]),score[key])
    check(f'{row.task} {row.method} {row.algorithm} dates',saved.target_time.dt.year.eq(2025).all() and (saved.target_time-saved.timestamp).eq(pd.Timedelta(hours=1)).all(),'one hour, 2025 targets')
summary=[];bootstrap=[]
for _,winner in selected.iterrows():
    task=winner.task;method=winner.method;name=winner.algorithm
    model_file=OUT/f'{task}_{method}_{name}.joblib'
    bundle=joblib.load(model_file)
    # Loaded pipeline contains training statistics, scalers and estimator.
    pred=bundle['pipeline'].predict(allparts[features])[-len(te):]
    saved=pd.read_csv(OUT/f'predictions_{task}_{method}_{name}_Test_2025.csv.gz',parse_dates=['timestamp','target_time'])
    check(f'{task} serialized pipeline predictions',np.allclose(pred,saved.predicted),model_file.name)
    expected=te.target_NO2_next_hour.to_numpy() if task=='forecast' else (te.target_NO2_next_hour>bundle['threshold']).astype(int).to_numpy()
    check(f'{task} targets observed',np.allclose(expected,saved.observed),'original cleaned city NO2')
    metric=run.regscore(saved.observed,saved.predicted) if task=='forecast' else run.clscore(saved.observed,saved.predicted,saved.probability)
    row=results[(results.task==task)&(results.method==method)&(results.algorithm==name)&(results.subset=='Test_2025')].iloc[0]
    key='RMSE' if task=='forecast' else 'F1'
    check(f'{task} metrics recomputed',np.isclose(metric[key],row[key]),metric[key])
    summary.append(row.to_dict())
    pre=bundle['pipeline'].named_steps['preprocess']
    prefix=allparts[features].iloc[:100]
    changed=allparts[features].iloc[:200].copy();changed.iloc[100:]=changed.iloc[100:]*5+123
    check(f'{task} no future dependence',np.allclose(pre.transform(prefix),pre.transform(changed)[:100]),method)
    if task=='forecast':
        saved['model_sq']=(saved.observed-saved.predicted)**2
        saved['base_sq']=(saved.observed-saved.persistence)**2
        saved['block']=((saved.target_time-saved.target_time.min()).dt.total_seconds()//(7*86400)).astype(int)
        b=saved.groupby('block').agg(m=('model_sq','sum'),p=('base_sq','sum'),n=('model_sq','size')).to_numpy()
        rng=np.random.default_rng(42); idx=rng.integers(0,len(b),size=(2000,len(b))); a=b[idx].sum(axis=1)
        gains=100*(1-np.sqrt(a[:,0]/a[:,1]))
        bootstrap.append(dict(method=method,algorithm=name,RMSE_gain_pct=100*(1-np.sqrt(saved.model_sq.mean()/saved.base_sq.mean())),CI95_low=np.quantile(gains,.025),CI95_high=np.quantile(gains,.975),block_days=7,replicates=2000))
# Explicit causal regression check even when another method wins validation.
sample=tr[features].iloc[:300].copy();sample.iloc[20:23,0]=np.nan
pre=PollutionPreprocessor('causal_linear').fit(sample)
modified=sample.copy();modified.iloc[150:]=12345
check('causal_linear prefix invariance',np.allclose(pre.transform(sample)[:150],pre.transform(modified)[:150]),'future feature modifications do not change past transformed values')
pd.DataFrame(checks).to_csv(OUT/'verification_checks.csv',index=False)
if any(c['status']=='FAIL' for c in checks):raise RuntimeError('Corrected verification failed: '+str(checks))
pd.DataFrame(summary).to_csv(OUT/'selected_test_results.csv',index=False)
pd.DataFrame(bootstrap).to_csv(OUT/'selected_forecast_bootstrap.csv',index=False)

for task,metric,title in [('forecast','RMSE','One-hour NO2 forecast: 2025 RMSE'),('classification','F1','Relative high-NO2 classification: 2025 F1')]:
    subset=results[(results.task==task)&(results.subset=='Test_2025')&(results.algorithm!='Persistence')]
    table=subset.pivot(index='method',columns='algorithm',values=metric).reindex(run.METHODS)
    fig,ax=plt.subplots(figsize=(10,4))
    im=ax.imshow(table.to_numpy(),cmap='viridis_r' if task=='forecast' else 'viridis',aspect='auto')
    ax.set_xticks(range(len(table.columns)),table.columns,rotation=20,ha='right');ax.set_yticks(range(len(table.index)),table.index)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):ax.text(j,i,f'{table.iloc[i,j]:.5f}',ha='center',va='center',color='white',fontsize=9)
    baseline=results[(results.task==task)&(results.algorithm=='Persistence')&(results.subset=='Test_2025')][metric].iloc[0]
    ax.set_title(f'{title}; persistence {baseline:.5f}');fig.colorbar(im,ax=ax,label=metric);fig.tight_layout()
    fig.savefig(FIG/f'{task}_grid.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4))
for scenario in ['primary','excluded_zeros']:
    frame=pd.read_csv(OUT/'data'/f'city_{scenario}.csv.gz',parse_dates=['timestamp']).set_index('timestamp')
    frame.NO2.resample('MS').mean().plot(ax=ax,label=scenario)
ax.set(ylabel='NO2 (source units)',title='Monthly city averages: sensitivity to all-zero records');ax.legend();fig.tight_layout();fig.savefig(FIG/'zero_sensitivity_trend.png',dpi=300);plt.close(fig)

limitations=[
    'Physical station identities and measurement units still require verification with the data owner. Fixed-network analysis uses persistent source labels, not confirmed physical identities.',
    'All-core-zero readings remain ambiguous. Primary analysis retains them; exclusion is a separate sensitivity scenario. It is not established that these records are instrument failures.',
    'Repeated timestamps with differing measurement vectors are averaged after exact repeat removal. Current counts are recorded in source_cleanup.csv.',
    'Manual secondary blocks are preserved as secondary_block fields and excluded from concentration features. Manual data are exported but not pooled into hourly predictive targets. No manual 2025 files were supplied.',
    'KNN uses a deterministic sample of up to 2,000 training-only donor rows, standardized using training statistics. This is a computational approximation and its donor sampling sensitivity is not evaluated.',
    'The causal linear method is past-only extrapolation for at most three hours with training-median fallback. At discontinuous times its history resets. Deployment must supply chronologically ordered feature rows and recent context.',
    'One validation year chooses models and one final year evaluates them. This does not establish performance in other cities, later years or longer forecasting horizons.',
    'PCA and clusters are exploratory; anomaly flags lack confirmed ground truth. Clustering uses station-average profiles, where chronological extrapolation is not defined; the causal method is therefore not applied to clustering.',
    'Sensitivity scenarios use different available stations and observed targets, so cross-scenario metric differences are not paired algorithm superiority tests.'
]
report=['# Исправленный pipeline: Алматы, 2021–2025','',f'Выполнено {len(checks)} проверок, все прошли. Сравнено 20 прогнозных сочетаний и 20 классификационных сочетаний. Выбор сделан по 2024 году, итоговая оценка — по 2025 году.','',pd.DataFrame(summary).to_string(index=False),'',pd.DataFrame(bootstrap).to_string(index=False),'','## Исправлено','',
    '- Точные повторы удалены, конфликтующие записи усреднены по уникальным векторам; текущие количества находятся в source_cleanup.csv.',
    '- Строки без первичных ручных концентраций исключены из измерительной выборки; вторичные блоки сохранены отдельно.',
    '- Все прогнозные импутеры, нормализация и доноры обучены только на 2021–2023. Оценка целей проводится на наблюдаемом NO2.',
    '- Аномалии теперь используют обучение preprocessing только на прошлом; станционно-календарная модель использует только наблюдаемые цели.',
    '- Для классификации добавлен persistence. Для каждой импутации сравниваются пять алгоритмов прогноза и пять классификаторов.',
    '- Выполнены сценарии исключения подозрительных нулей и фиксированной сети доступных названий.',
    '- Модели сохранены вместе с preprocessing; воспроизведение из joblib проверено.',
    '- Почасовые сетки станций непрерывны; разрывы времени сбрасывают причинную историю.',
    '','## Остающиеся ограничения','']+[f'- {x}' for x in limitations]
(OUT/'CORRECTION_REPORT_RU.md').write_text('\n'.join(report),encoding='utf-8')
baseline_rows=results[(results.subset=='Test_2025')&(results.algorithm=='Persistence')]
baseline_fore=baseline_rows[baseline_rows.task=='forecast'].iloc[0]
baseline_cla=baseline_rows[baseline_rows.task=='classification'].iloc[0]
with (OUT/'CORRECTION_REPORT_RU.md').open('a',encoding='utf-8') as handle:
    handle.write(f'\n\n## Сравнение с базовой моделью\n\nВыбранный по валидации прогноз на тесте немного хуже persistence: RMSE {summary[0]["RMSE"]:.6f} против {baseline_fore.RMSE:.6f}. Выбранный классификатор: F1 {summary[1]["F1"]:.6f} против {baseline_cla.F1:.6f}. В сценариях исключения нулей и фиксированной сети выбранные модели также не превосходят persistence по основным метрикам. Устойчивого преимущества машинного обучения не установлено. Нельзя менять победителя по результатам теста; дальнейшее улучшение требует новых признаков и новой независимой проверки.\n')
manifest['verification']={'passed':len(checks),'failed':0,'status':'corrected_outputs_verified_with_source_limitations'}
manifest['clustering_imputers']=['mean','median','KNN']
with pd.ExcelWriter(OUT/'corrected_comparison.xlsx') as writer:
    for sheet,file in [('Predictive models','predictive_comparison.csv'),('Validation selection','validation_selected_models.csv'),('Sensitivity','sensitivity_comparison.csv'),('Exploratory','exploratory_comparison.csv'),('Source cleanup','source_cleanup.csv'),('Verification','verification_checks.csv'),('Forecast uncertainty','selected_forecast_bootstrap.csv')]:
        pd.read_csv(OUT/file).to_excel(writer,sheet_name=sheet,index=False)
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(bootstrap).to_string(index=False));print('FINALIZED AND VERIFIED',len(checks),'checks')
