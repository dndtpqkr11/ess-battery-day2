"""DAY 2 regression experiment. Does not write any DAY 1 file.

All model/feature/target choices use grouped CV on the development train split.
Hold-out and external batches are evaluated only after selection is frozen.
"""
from pathlib import Path
import os, json, hashlib
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mini-project-day2-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn, joblib
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from scipy.special import exp10

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
SEED=42
SETS={
 'delta_only':['log_dq_var'],
 'compact':['log_dq_var','QD_slope_100','chargetime_mean_100','Tavg_mean_100','IR_mean_100'],
 'compact_charge':['log_dq_var','charge_I_time_std_5','QD_slope_100','chargetime_mean_100','Tavg_mean_100','IR_mean_100']}

def make(model,log_target):
    pipe=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler()),('model',model)])
    return TransformedTargetRegressor(regressor=pipe,func=np.log10 if log_target else None,inverse_func=exp10 if log_target else None)

def metrics(y,p):
    return {'MAPE_pct':float(mean_absolute_percentage_error(y,p)*100),'MAE_cycles':float(mean_absolute_error(y,p)),
            'RMSE_cycles':float(np.sqrt(mean_squared_error(y,p))),'n':len(y),'nonpositive_predictions':int(np.sum(p<=0))}

def verify_day1():
    # Historical preview snapshot is retained. This guard protects the submitted version.
    snapshot_path=HERE/'submitted_day1_snapshot.json'
    if not snapshot_path.exists():
        return 0  # Standalone clone has no DAY 1 parent workspace.
    snap=json.loads(snapshot_path.read_text())
    changed=[name for name,digest in snap.items() if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    if changed: raise RuntimeError(f'DAY 1 files changed: {changed}')
    return len(snap)

def run():
    verify_day1()
    for path in ['data','results','figures','models']: (HERE/path).mkdir(exist_ok=True)
    d=pd.read_csv(HERE/'data/features.csv')
    base=d
    all_features=list(dict.fromkeys(f for fs in SETS.values() for f in fs))
    # Explicit allowlist: future fade, knee and cycle counts are never features.
    d=d[['cell_id','batch','policy','cycle_life','possible_censored',*all_features]].copy()
    if d.cell_id.duplicated().any() or d.possible_censored.dtype!=bool:
        raise ValueError('Invalid cell IDs or measurement-end flags')
    if d.batch.value_counts().to_dict()!={1:46,2:47,3:46}:
        raise ValueError('Unexpected dataset composition')
    eligible=d[(d.batch==1)&~d.possible_censored&d.cycle_life.notna()&(d.cycle_life>0)].reset_index(drop=True)
    splitter=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED)
    ti,vi=next(splitter.split(eligible,groups=eligible.policy)); train=eligible.iloc[ti].copy(); valid=eligible.iloc[vi].copy()
    assert set(train.cell_id).isdisjoint(valid.cell_id) and set(train.policy).isdisjoint(valid.policy)
    cv=list(GroupKFold(n_splits=3).split(train,groups=train.policy))
    for a,b in cv:
        assert set(train.iloc[a].policy).isdisjoint(train.iloc[b].policy)
    split_rows=[]
    for label,g in [('train',train),('valid',valid)]:
        for _,r in g.iterrows(): split_rows.append({'cell_id':r.cell_id,'split':label,'policy':r.policy})
    pd.DataFrame(split_rows).to_csv(HERE/'results/split.csv',index=False)
    fold_rows=[{'fold':i+1,'train_n':len(a),'valid_n':len(b),'train_ids':train.iloc[a].cell_id.tolist(),'valid_ids':train.iloc[b].cell_id.tolist()} for i,(a,b) in enumerate(cv)]
    (HERE/'results/cv_folds.json').write_text(json.dumps(fold_rows,indent=2))
    print(f'Train={len(train)} ({train.policy.nunique()} groups), Valid={len(valid)} ({valid.policy.nunique()} groups)',flush=True)
    specs=[
        ('DummyMedian',DummyRegressor(strategy='median'),{}),
        ('Linear',LinearRegression(),{}),
        ('Ridge',Ridge(),{'regressor__model__alpha':[.1,1.,10.]}),
        ('ElasticNet',ElasticNet(max_iter=10000,random_state=SEED),{'regressor__model__alpha':[.01,.1,1.],'regressor__model__l1_ratio':[.2,.8]}),
        ('RandomForest',RandomForestRegressor(n_estimators=200,random_state=SEED,n_jobs=1),{'regressor__model__max_depth':[2,3],'regressor__model__min_samples_leaf':[2,4]})]
    best=None; records=[]; searches=[]; dummy=None
    for name,model,params in specs:
        sets=['delta_only'] if name=='DummyMedian' else list(SETS)
        targets=[False] if name=='DummyMedian' else [False,True]
        for feature_set in sets:
            for log_target in targets:
                fs=SETS[feature_set]; search=GridSearchCV(make(model,log_target),params or {},scoring='neg_mean_absolute_percentage_error',cv=cv,refit=True,n_jobs=1,error_score='raise')
                search.fit(train[fs],train.cycle_life)
                result=search.cv_results_; idx=search.best_index_; mean=-float(search.best_score_)*100
                record={'model':name,'feature_set':feature_set,'target':'log10' if log_target else 'raw','cv_MAPE_pct':mean,'cv_std_pct':float(result['std_test_score'][idx])*100,'params':json.dumps(search.best_params_,sort_keys=True)}
                records.append(record)
                for k,param in enumerate(result['params']):
                    searches.append({'model':name,'feature_set':feature_set,'target':record['target'],'params':json.dumps(param,sort_keys=True),'cv_MAPE_pct':-result['mean_test_score'][k]*100,
                                     **{f'fold{j+1}_MAPE_pct':-result[f'split{j}_test_score'][k]*100 for j in range(3)}})
                if name=='DummyMedian': dummy=search.best_estimator_
                elif best is None or mean<best['cv_MAPE_pct']:
                    best={**record,'features':fs,'estimator':search.best_estimator_}
        print(f'{name} CV search complete',flush=True)
    pd.DataFrame(records).sort_values('cv_MAPE_pct').to_csv(HERE/'results/candidates.csv',index=False)
    pd.DataFrame(searches).to_csv(HERE/'results/search_results.csv',index=False)
    selected={k:v for k,v in best.items() if k!='estimator'}
    selected.update({'selection_rule':'Minimum mean grouped CV MAPE on development train only','seed':SEED,'sklearn_version':sklearn.__version__,'input_sha256':hashlib.sha256((HERE/'data/features.csv').read_bytes()).hexdigest(),'train_ids':train.cell_id.tolist(),'valid_ids':valid.cell_id.tolist()})
    (HERE/'results/selected_model.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    # Freeze/save before accessing any hold-out or external evaluation labels.
    joblib.dump({'model':best['estimator'],'features':best['features'],'selection':selected},HERE/'models/selected_model.joblib')
    eval_sets={'Valid_B1':valid,'Test_B2':d[(d.batch==2)&d.cycle_life.notna()&(d.cycle_life>0)],'Additional_B3':d[(d.batch==3)&d.cycle_life.notna()&(d.cycle_life>0)]}
    eval_rows=[]; predictions=[]
    for label,g in eval_sets.items():
        for name,est,fs in [('Selected',best['estimator'],best['features']),('DummyMedian',dummy,SETS['delta_only'])]:
            pred=est.predict(g[fs]); metric=metrics(g.cycle_life.to_numpy(),pred)
            eval_rows.append({'model':name,'split':label,**metric})
            for cid,y,p in zip(g.cell_id,g.cycle_life,pred): predictions.append({'model':name,'split':label,'cell_id':cid,'actual':y,'predicted':p,'APE_pct':abs(y-p)/y*100})
    evaluations=pd.DataFrame(eval_rows); evaluations.to_csv(HERE/'results/evaluations.csv',index=False)
    pd.DataFrame(predictions).to_csv(HERE/'results/predictions.csv',index=False)
    # Targets missing from source are predicted separately and never scored or imputed.
    missing=d[(d.batch.isin([2,3]))&d.cycle_life.isna()].copy()
    missing['predicted_cycle_life']=best['estimator'].predict(missing[best['features']])
    missing[['cell_id','batch','predicted_cycle_life']].to_csv(HERE/'results/missing_target_predictions.csv',index=False)
    chosen=evaluations[evaluations.model=='Selected'].set_index('split')
    vm=float(chosen.loc['Valid_B1','MAPE_pct']); tm=float(chosen.loc['Test_B2','MAPE_pct'])
    reporting={'Model':best['model'],'Features':best['feature_set'],'Train_B1_CV_MAPE_pct':best['cv_MAPE_pct'],'Train_B1_CV_std_pct':best['cv_std_pct'],'Valid_B1_MAPE_pct':vm,'Test_B2_MAPE_pct':tm,
               'Gap_Train_Valid_pp':vm-best['cv_MAPE_pct'],'Gap_Valid_Test_pp':tm-vm,'Gap_Target_Test_pp':tm-9.1,'Target_MAPE_pct':9.1,
               'Gap_formula':'Valid-Train; Test-Valid; Test-Target (positive = larger error)',
               'Additional_B3_MAPE_pct':float(chosen.loc['Additional_B3','MAPE_pct'])}
    pd.DataFrame([reporting]).to_csv(HERE/'results/performance_reporting.csv',index=False)
    # Independent formula cross-check against sklearn for every scored prediction.
    pp=pd.DataFrame(predictions)
    for _,r in evaluations.iterrows():
        a=pp[(pp.model==r.model)&(pp.split==r.split)]
        assert np.isclose(np.mean(np.abs(a.actual-a.predicted)/a.actual)*100,r.MAPE_pct)
        assert np.isclose(np.mean(np.abs(a.actual-a.predicted)),r.MAE_cycles)
    loaded=joblib.load(HERE/'models/selected_model.joblib')
    assert np.allclose(loaded['model'].predict(valid[loaded['features']]),best['estimator'].predict(valid[best['features']]))
    actual_pipe=best['estimator'].regressor_; fitted=actual_pipe.named_steps['model']
    if hasattr(fitted,'coef_'):
        imp=pd.DataFrame({'feature':best['features'],'standardized_coefficient':fitted.coef_})
    else:
        imp=pd.DataFrame({'feature':best['features'],'importance':fitted.feature_importances_})
    imp.to_csv(HERE/'results/model_feature_effects.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,(label,g) in zip(axs,pp[pp.model=='Selected'].groupby('split',sort=False)):
        ax.scatter(g.actual,g.predicted,alpha=.7); limits=[min(g.actual.min(),g.predicted.min())*.9,max(g.actual.max(),g.predicted.max())*1.05]
        ax.plot(limits,limits,'--',color='gray'); ax.set(title=label,xlabel='Actual cycle life',ylabel='Predicted cycle life',xlim=limits,ylim=limits)
    fig.tight_layout(); fig.savefig(HERE/'figures/prediction_scatter.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4)); ordered=pd.DataFrame(records).sort_values('cv_MAPE_pct').head(12)
    labels=ordered.model+' / '+ordered.feature_set+' / '+ordered.target
    ax.barh(labels[::-1],ordered.cv_MAPE_pct[::-1],color='#2563eb'); ax.set_xlabel('Grouped CV MAPE (%)'); fig.tight_layout(); fig.savefig(HERE/'figures/cv_comparison.png',dpi=170); plt.close(fig)
    counts={'raw_by_batch':base.groupby('batch').size().to_dict(),'input_sha256':selected['input_sha256'],'B1_end_warning_excluded':base[(base.batch==1)&base.possible_censored].cell_id.tolist(),
            'missing_target_excluded':base[base.cycle_life.isna()].cell_id.tolist(),'eligible_B1':len(eligible),'train':len(train),'valid':len(valid),'train_policy_groups':train.policy.nunique(),'valid_policy_groups':valid.policy.nunique(),'candidate_best_rows':len(records),'total_cv_configs':len(searches),'day1_unchanged_files':verify_day1()}
    (HERE/'results/validation.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2))
    print(json.dumps(reporting,ensure_ascii=False,indent=2),flush=True)
    print(evaluations.to_string(index=False),flush=True)

if __name__=='__main__': run()
