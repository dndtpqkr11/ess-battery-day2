"""Extract early-cycle predictors from the three assignment MAT files.

Full-trace information is used only to flag questionable training labels, never
as a predictor. Outputs stay in DAY 2; no DAY 1 scripts or artifacts are needed.
"""
from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
FEATURES=['log_dq_var','QD_slope_100','chargetime_mean_100','Tavg_mean_100',
          'IR_mean_100','charge_I_time_std_5']

def extract(raw_dir):
    rows=[]
    for batch,date in enumerate(['2017-05-12','2018-02-20','2018-04-12'],1):
        matches=list(raw_dir.glob(date+'*.mat'))
        if len(matches)!=1:
            raise ValueError(f'Expected one {date} MAT file; found {len(matches)}')
        with h5py.File(matches[0],'r') as f:
            b=f['batch']
            for i in range(b['cycle_life'].size):
                s=f[b['summary'][i,0]]; c=f[b['cycles'][i,0]]
                x=s['cycle'][()].ravel(); q=s['QDischarge'][()].ravel()
                life=float(f[b['cycle_life'][i,0]][()].ravel()[0])
                idx={int(no):j for j,no in enumerate(x)}
                row={'cell_id':f'b{batch}c{i}','batch':batch,
                     'policy':''.join(chr(int(v)) for v in f[b['policy_readable'][i,0]][()].ravel()),
                     'cycle_life':life,'possible_censored':bool(q[-1]>.885 and life==len(x)+1)}
                for key in ['chargetime','Tavg','IR']:
                    value=s[key][()].ravel().copy()
                    if key=='IR':value[value<=0]=np.nan
                    value=value[(x>=2)&(x<=100)];value=value[np.isfinite(value)]
                    row[f'{key}_mean_100']=float(value.mean()) if len(value) else np.nan
                good=(x>=2)&(x<=100)&np.isfinite(q)&(q>0)&(q<1.3)
                row['QD_slope_100']=float(np.polyfit(x[good],q[good],1)[0]) if good.sum()>2 else np.nan
                if 10 not in idx or 100 not in idx:raise ValueError(f'Missing early cycle: {row["cell_id"]}')
                q10=f[c['Qdlin'][idx[10],0]][()].ravel();q100=f[c['Qdlin'][idx[100],0]][()].ravel()
                v=f[b['Vdlin'][i,0]][()].ravel()
                if len(q10)!=len(q100) or len(q10)!=len(v) or not np.isfinite(q100-q10).all():
                    raise ValueError(f'Invalid delta-Q: {row["cell_id"]}')
                row['log_dq_var']=float(np.log10(max(np.var(q100-q10),1e-15)))
                values=[];weights=[];present=[]
                for no in [2,3,4,5]:
                    if no not in idx:continue
                    current=f[c['I'][idx[no],0]][()].ravel();t=f[c['t'][idx[no],0]][()].ravel()
                    if len(current)!=len(t):raise ValueError('Current/time length mismatch')
                    dt=np.diff(t)
                    ok=np.isfinite(dt)&(dt>0)&np.isfinite(current[:-1])&np.isfinite(current[1:])&(current[:-1]>.1)&(current[1:]>.1)
                    interval=(current[:-1]+current[1:])/2
                    if ok.any():values.extend(interval[ok]);weights.extend(dt[ok]);present.append(no)
                row['charge_I_time_std_5']=np.nan
                if len(present)==4 and values:
                    a,w=np.array(values),np.array(weights);w=w/w.sum();mean=np.sum(a*w)
                    row['charge_I_time_std_5']=float(np.sqrt(np.sum(w*(a-mean)**2)))
                rows.append(row)
        count=sum(r['batch']==batch for r in rows)
        print(f'Batch {batch}: {count} cells',flush=True)
    return pd.DataFrame(rows)[['cell_id','batch','policy','cycle_life','possible_censored',*FEATURES]]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=HERE/'data/features.csv')
    args=parser.parse_args();data=extract(args.raw_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True);data.to_csv(args.output,index=False)
    print(args.output)

if __name__=='__main__':main()
