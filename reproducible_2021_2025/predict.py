"""Predict a chronological CSV containing timestamp and core concentrations.

Example: python predict_corrected.py --model results_corrected/forecast_mean_ExtraTrees.joblib
 --input new_station_hourly.csv --output predictions.csv

Input must use the same verified units and aggregation as training. For causal
methods include recent history before the prediction window. No model refit occurs.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import features as base

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args()
    frame=pd.read_csv(args.input,parse_dates=['timestamp'])
    if frame.timestamp.isna().any():raise ValueError('Invalid timestamps')
    for col in base.CORE:
        if col not in frame:frame[col]=float('nan')
        frame[col]=pd.to_numeric(frame[col],errors='coerce')
    city=base.city_series(frame.sort_values('timestamp'))
    lagged,_=base.add_lags(city);lagged=lagged.set_index('timestamp',drop=False)
    bundle=joblib.load(args.model)
    pred=bundle['pipeline'].predict(lagged[bundle['features']])
    output=pd.DataFrame({'timestamp':lagged.timestamp,'target_time':lagged.target_time,'prediction':pred})
    if hasattr(bundle['pipeline'],'predict_proba'):
        output['high_NO2_probability']=bundle['pipeline'].predict_proba(lagged[bundle['features']])[:,1]
        output['training_NO2_threshold']=bundle['threshold']
    output.to_csv(args.output,index=False)
    print(f'Saved {len(output)} predictions to {Path(args.output).resolve()}')

if __name__=='__main__':main()
