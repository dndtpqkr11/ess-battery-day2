"""Predict total cycle life from an early-feature CSV using the saved model."""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=HERE/'models/improved_model.joblib')
    parser.add_argument('--diagnostics',action='store_true',help='Include training-range flags; these are not confidence intervals')
    args=parser.parse_args()
    artifact=joblib.load(args.model);data=pd.read_csv(args.input)
    missing=set(artifact['features'])-set(data)
    if missing:raise ValueError(f'Missing required features: {sorted(missing)}')
    x=data[artifact['features']].apply(pd.to_numeric,errors='raise')
    if np.isinf(x.to_numpy()).any():raise ValueError('Infinite predictor value')
    prediction=artifact['model'].predict(x)
    if not np.isfinite(prediction).all() or (prediction<=0).any():raise ValueError('Invalid predicted cycle life')
    out=data[['cell_id']].copy() if 'cell_id' in data else pd.DataFrame({'row':range(len(data))})
    out['predicted_cycle_life']=prediction
    if args.diagnostics:
        ranges=artifact['selection'].get('feature_ranges')
        target_range=artifact['selection'].get('training_target_range')
        if not ranges or not target_range:
            raise ValueError('This model has no range metadata; use improved_model.joblib for diagnostics')
        flags=[]
        for f in artifact['features']:
            flag=(x[f]<ranges[f]['min'])|(x[f]>ranges[f]['max'])
            out[f'{f}_outside_train_range']=flag
            flags.append(flag.to_numpy())
        out['missing_feature_count']=x.isna().sum(axis=1)
        out['any_feature_outside_train_range']=np.any(flags,axis=0)
        out['prediction_outside_train_target_range']=(prediction<target_range[0])|(prediction>target_range[1])
    args.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(args.output,index=False)
    print(f'{len(out)} predictions -> {args.output}')

if __name__=='__main__':main()
