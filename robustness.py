"""Follow-up feature ablation and policy-split stability on B1 train only.

Primary results are immutable. B2/B3 scores are reused-test diagnostics, not a
fresh independent evaluation. No external score enters candidate selection.
"""
from pathlib import Path
import json,hashlib,os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mini-project-day2-mpl')
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from train import make,metrics,verify_day1

HERE=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    verify_day1()
    protected=[HERE/'models/selected_model.joblib',HERE/'results/predictions.csv',HERE/'results/performance_reporting.csv']
    before={str(p.relative_to(HERE)):digest(p) for p in protected}
    d=pd.read_csv(HERE/'data/features.csv');original=joblib.load(protected[0]);meta=original['selection']
    train=d.set_index('cell_id',drop=False).loc[meta['train_ids']].reset_index(drop=True)
    valid=d[d.cell_id.isin(meta['valid_ids'])]
    fs=original['features']
    specs={
        'original_six':(fs,make(Ridge(alpha=1),True)),
        'drop_current_std':([f for f in fs if f!='charge_I_time_std_5'],make(Ridge(alpha=1),True)),
        'drop_IR':([f for f in fs if f!='IR_mean_100'],make(Ridge(alpha=1),True)),
        'delta_current':(['log_dq_var','charge_I_time_std_5'],make(Ridge(alpha=1),True)),
        'delta_slope':(['log_dq_var','QD_slope_100'],make(Ridge(alpha=1),True)),
        'delta_ridge_log':(['log_dq_var'],make(Ridge(alpha=1),True)),
        'delta_linear_raw':(['log_dq_var'],make(LinearRegression(),False))}
    protocol={'purpose':'B1 train-only ablation and repeated policy hold-out stability',
              'train_ids':meta['train_ids'],'excluded_original_holdout_ids':meta['valid_ids'],
              'seeds':list(range(20)),'group_holdout_fraction':.25,'candidates':{k:v[0] for k,v in specs.items()},
              'selection_rule':'Minimum mean MAPE across the 20 fixed repeated policy hold-outs; tie by fewer features then name',
              'external_evaluation':'Already-seen B2/B3; diagnostic reuse, not a new untouched test'}
    (HERE/'results/followup_protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];folds=[];coefs=[]
    for seed in protocol['seeds']:
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(train,groups=train.policy))
        t,v=train.iloc[a],train.iloc[b]
        assert set(t.policy).isdisjoint(v.policy)
        assert set(t.cell_id)|set(v.cell_id)==set(meta['train_ids'])
        folds.append({'seed':seed,'train_ids':t.cell_id.tolist(),'valid_ids':v.cell_id.tolist()})
        for name,(features,est) in specs.items():
            fitted=clone(est).fit(t[features],t.cycle_life);prediction=fitted.predict(v[features])
            rows.append({'candidate':name,'seed':seed,'train_n':len(t),'valid_n':len(v),**metrics(v.cycle_life.to_numpy(),prediction)})
            if name=='original_six':
                coefs.extend({'seed':seed,'feature':f,'coefficient':float(c)} for f,c in zip(features,fitted.regressor_.named_steps['model'].coef_))
    scores=pd.DataFrame(rows);scores.to_csv(HERE/'results/policy_stability_scores.csv',index=False)
    (HERE/'results/policy_stability_splits.json').write_text(json.dumps(folds,indent=2))
    summary=scores.groupby('candidate').MAPE_pct.agg(mean_MAPE_pct='mean',median_MAPE_pct='median',std_MAPE_pct='std',min_MAPE_pct='min',max_MAPE_pct='max').reset_index()
    summary['n_features']=summary.candidate.map({k:len(v[0]) for k,v in specs.items()})
    summary=summary.sort_values(['mean_MAPE_pct','n_features','candidate']);summary.to_csv(HERE/'results/policy_stability_summary.csv',index=False)
    wide=scores.pivot(index='seed',columns='candidate',values='MAPE_pct')
    paired=pd.DataFrame({'candidate':wide.columns,'mean_change_vs_original_pp':[(wide[c]-wide.original_six).mean() for c in wide],
        'better_split_count':[int((wide[c]<wide.original_six).sum()) for c in wide],'split_count':20})
    paired.to_csv(HERE/'results/ablation_paired.csv',index=False)
    coeff=pd.DataFrame(coefs);coeff.to_csv(HERE/'results/coefficient_stability.csv',index=False)
    coeff.groupby('feature').coefficient.agg(mean='mean',minimum='min',maximum='max',positive_count=lambda a:int((a>0).sum()),negative_count=lambda a:int((a<0).sum())).to_csv(HERE/'results/coefficient_stability_summary.csv')
    chosen=summary.iloc[0];name=chosen.candidate;features,est=specs[name]
    followup=clone(est).fit(train[features],train.cycle_life)
    fixed_cv=[]
    indexed=train.set_index('cell_id')
    for fold in json.loads((HERE/'results/cv_folds.json').read_text()):
        a,b=indexed.loc[fold['train_ids']],indexed.loc[fold['valid_ids']]
        fitted=clone(est).fit(a[features],a.cycle_life)
        fixed_cv.append(metrics(b.cycle_life.to_numpy(),fitted.predict(b[features]))['MAPE_pct'])
    choice={'candidate':name,'features':features,'mean_repeated_holdout_MAPE_pct':float(chosen.mean_MAPE_pct),
            'fixed_cv_MAPE_pct':float(np.mean(fixed_cv)),'fixed_cv_std_pct':float(np.std(fixed_cv)),
            'fixed_cv_fold_MAPE_pct':fixed_cv,'selection_rule':protocol['selection_rule'],'train_ids':meta['train_ids'],'input_sha256':digest(HERE/'data/features.csv')}
    (HERE/'results/followup_selected.json').write_text(json.dumps(choice,indent=2))
    joblib.dump({'model':followup,'features':features,'selection':choice},HERE/'models/followup_model.joblib')
    # External labels are accessed only after B1-only selection has been saved.
    evaluations=[];predictions=[]
    for split,g in [('Valid_B1_reused',valid),('Test_B2_reused',d[(d.batch==2)&d.cycle_life.notna()]),('Additional_B3_reused',d[(d.batch==3)&d.cycle_life.notna()])]:
        prediction=followup.predict(g[features]);evaluations.append({'candidate':name,'split':split,**metrics(g.cycle_life.to_numpy(),prediction)})
        predictions.extend({'split':split,'cell_id':cid,'actual':float(y),'predicted':float(p),'APE_pct':float(abs(y-p)/y*100)} for cid,y,p in zip(g.cell_id,g.cycle_life,prediction))
    pd.DataFrame(evaluations).to_csv(HERE/'results/followup_evaluations.csv',index=False)
    cohort_path=HERE/'results/B3_quality_cohort.csv'
    if cohort_path.exists():
        cohort=pd.read_csv(cohort_path);g=d[d.cell_id.isin(cohort[cohort.retained].cell_id)&d.cycle_life.notna()]
        sensitivity=[]
        for model_name,model,features0 in [('preview_Ridge',original['model'],fs),('followup_Linear',followup,features)]:
            sensitivity.append({'model':model_name,'cohort':'Published-rule B3 subset, reused diagnostic',**metrics(g.cycle_life.to_numpy(),model.predict(g[features0]))})
        pd.DataFrame(sensitivity).to_csv(HERE/'results/B3_quality_sensitivity.csv',index=False)
    evaluation=pd.DataFrame(evaluations).set_index('split')
    vm=float(evaluation.loc['Valid_B1_reused','MAPE_pct']);tm=float(evaluation.loc['Test_B2_reused','MAPE_pct'])
    pd.DataFrame([
        ['Train (Batch 1 CV)',choice['fixed_cv_MAPE_pct'],'Original fixed 3-fold; diagnostic after follow-up selection'],
        ['Valid (Batch 1 Hold-out)',vm,'Previously seen; excluded from follow-up selection'],
        ['Test (Batch 2)',tm,'Previously seen; reused-test score'],
        ['Gap (Train-Valid)',vm-choice['fixed_cv_MAPE_pct'],'Valid-Train; percentage points'],
        ['Gap (Valid-Test)',tm-vm,'Test-Valid; percentage points'],
        ['Gap (Target-Test)',tm-9.1,'Test-9.1; percentage points']
    ],columns=['구분','MAPE (%) / Gap (%p)','비고']).to_csv(HERE/'results/followup_performance.csv',index=False)
    pd.DataFrame(predictions).to_csv(HERE/'results/followup_predictions.csv',index=False)
    missing=d[d.batch.isin([2,3])&d.cycle_life.isna()].copy()
    missing['predicted_cycle_life']=followup.predict(missing[features])
    missing[['cell_id','batch','predicted_cycle_life']].to_csv(HERE/'results/followup_missing_target_predictions.csv',index=False)
    final_pred=pd.DataFrame(predictions)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,(split,g) in zip(axes,final_pred.groupby('split',sort=False)):
        ax.scatter(g.actual,g.predicted,color='#16a34a',alpha=.7)
        lo=min(g.actual.min(),g.predicted.min())*.9;hi=max(g.actual.max(),g.predicted.max())*1.05
        ax.plot([lo,hi],[lo,hi],'--',color='gray')
        score=float(evaluation.loc[split,'MAPE_pct'])
        ax.set(title=f'{split}\nMAPE={score:.2f}%',xlabel='Actual cycle life',ylabel='Predicted cycle life',xlim=(lo,hi),ylim=(lo,hi))
    fig.tight_layout();fig.savefig(HERE/'figures/final_prediction_scatter.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    labels=summary.candidate.tolist();ax.boxplot([wide[c].to_numpy() for c in labels],orientation='horizontal',tick_labels=labels)
    ax.set(xlabel='B1 train repeated policy hold-out MAPE (%)',title='20 fixed policy splits; original hold-out excluded')
    fig.tight_layout();fig.savefig(HERE/'figures/policy_stability.png',dpi=170);plt.close(fig)
    # Exact additive contributions in log10 prediction space, not causal effects.
    b2=d[(d.batch==2)&d.cycle_life.notna()];pipe=original['model'].regressor_
    z=pipe.named_steps['scale'].transform(pipe.named_steps['impute'].transform(b2[fs]));linear=pipe.named_steps['model']
    contributions=z*linear.coef_
    np.testing.assert_allclose(10**(linear.intercept_+contributions.sum(axis=1)),original['model'].predict(b2[fs]),rtol=1e-10)
    c=pd.DataFrame(contributions,columns=fs);c.insert(0,'cell_id',b2.cell_id.to_numpy());c['actual']=b2.cycle_life.to_numpy();c['predicted']=original['model'].predict(b2[fs]);c['intercept_log10']=linear.intercept_
    c.to_csv(HERE/'results/batch2_prediction_contributions.csv',index=False)
    issue=b2[['cell_id','cycle_life']].copy();issue['predicted']=c.predicted.to_numpy();issue['IR_missing']=b2.IR_mean_100.isna()
    for feature in fs:
        issue[f'{feature}_outside_train_range']=(b2[feature]<train[feature].min())|(b2[feature]>train[feature].max())
    issue['APE_pct']=abs(issue.predicted-issue.cycle_life)/issue.cycle_life*100
    issue.to_csv(HERE/'results/batch2_error_diagnostics.csv',index=False)
    diag=[]
    for name,mask in [('IR_missing',issue.IR_missing),('IR_observed',~issue.IR_missing),('charge_time_above_train_max',b2.chargetime_mean_100>train.chargetime_mean_100.max()),('charge_time_in_or_below_train_range',b2.chargetime_mean_100<=train.chargetime_mean_100.max())]:
        g=issue[mask];diag.append({'group':name,'n':len(g),'MAPE_pct':float(g.APE_pct.mean()),'mean_signed_error_cycles':float((g.predicted-g.cycle_life).mean())})
    pd.DataFrame(diag).to_csv(HERE/'results/error_subgroups.csv',index=False)
    assert all(digest(HERE/k)==v for k,v in before.items())
    audit={'status':'PASS','primary_artifacts_sha256':before,'primary_results_unchanged':True,
           'selection_used_B1_train_only':True,'original_holdout_used_for_selection':False,
           'external_scores_are_reused_test_diagnostics':True,'policy_disjoint_splits':20,
           'submitted_day1_unchanged_files':verify_day1()}
    (HERE/'results/followup_validation.json').write_text(json.dumps(audit,indent=2))
    print(summary.to_string(index=False));print(pd.DataFrame(evaluations).to_string(index=False));print(pd.DataFrame(diag).to_string(index=False))

if __name__=='__main__':run()
