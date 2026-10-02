"""Independently refit improvement scores and verify the weighted LAD optimum."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from improve import specs, FEATURES
from train import verify_day1

HERE=Path(__file__).resolve().parent


def refit(name, data):
    estimator, power=specs()[name]
    kwargs={}
    if power:
        w=1/np.power(data.cycle_life.to_numpy(),power)
        kwargs['model__sample_weight']=w/w.mean()
    return clone(estimator).fit(data[FEATURES],data.cycle_life,**kwargs)


def audit_scores(data, scores, seeds):
    assert len(scores)==len(seeds)*len(specs())
    assert not scores.duplicated(['seed','candidate']).any()
    for seed in seeds:
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(data,groups=data.policy))
        t,v=data.iloc[a],data.iloc[b]
        assert set(t.policy).isdisjoint(v.policy) and set(t.cell_id).isdisjoint(v.cell_id)
        for name in specs():
            p=refit(name,t).predict(v[FEATURES])
            stored=scores[(scores.seed==seed)&(scores.candidate==name)].iloc[0]
            np.testing.assert_allclose(stored.MAPE_pct,np.mean(np.abs(p-v.cycle_life)/v.cycle_life)*100,rtol=1e-8)
    return scores.groupby('candidate').MAPE_pct.mean().sort_values()


def verify():
    d=pd.read_csv(HERE/'data/features.csv').set_index('cell_id',drop=False)
    protocol=json.loads((HERE/'results/improvement_protocol.json').read_text())
    meta=json.loads((HERE/'results/improvement_selected.json').read_text())
    assert hashlib.sha256((HERE/'data/features.csv').read_bytes()).hexdigest()==meta['input_sha256']
    for path,digest in protocol['protected_sha256'].items():
        assert hashlib.sha256((HERE/path).read_bytes()).hexdigest()==digest,path
    original=json.loads((HERE/'results/selected_model.json').read_text())
    assert meta['train_ids']==original['train_ids']
    assert meta['valid_ids']==original['valid_ids']
    train=d.loc[meta['train_ids']]
    assert set(train.policy).isdisjoint(d.loc[meta['valid_ids']].policy)
    assert (train.batch==1).all() and not train.possible_censored.any()
    scores=pd.read_csv(HERE/'results/improvement_scores.csv')
    means=audit_scores(train,scores,protocol['seeds'])
    assert means.index[0]==meta['candidate']
    np.testing.assert_allclose(means.iloc[0],meta['mean_repeated_holdout_MAPE_pct'])
    nested=json.loads((HERE/'results/improvement_nested_folds.json').read_text())
    inner=pd.read_csv(HERE/'results/improvement_nested_search.csv')
    npred=pd.read_csv(HERE/'results/improvement_nested_predictions.csv')
    assert npred.cell_id.is_unique and set(npred.cell_id)==set(train.cell_id)
    nested_scores=[]
    for f in nested:
        t,v=d.loc[f['train_ids']],d.loc[f['valid_ids']]
        assert set(t.cell_id)|set(v.cell_id)==set(train.cell_id)
        assert set(t.cell_id).isdisjoint(v.cell_id) and set(t.policy).isdisjoint(v.policy)
        ranked=audit_scores(t,inner[inner.outer_fold==f['fold']],protocol['seeds'])
        assert ranked.index[0]==f['selected']
        p=refit(f['selected'],t).predict(v[FEATURES])
        saved=npred.set_index('cell_id').loc[v.cell_id]
        np.testing.assert_allclose(saved.predicted,p,rtol=1e-8)
        np.testing.assert_allclose(saved.previous_predicted,refit('previous_linear',t).predict(v[FEATURES]))
        value=np.mean(np.abs(p-v.cycle_life)/v.cycle_life)*100
        np.testing.assert_allclose(value,f['MAPE_pct'])
        nested_scores.append(value)
    np.testing.assert_allclose(np.mean(nested_scores),meta['nested_cv_MAPE_pct'])
    saved=joblib.load(HERE/'models/improved_model.joblib')
    assert saved['selection']==meta and saved['features']==FEATURES
    fitted=refit(meta['candidate'],train)
    pipe=saved['model'].regressor_
    np.testing.assert_allclose(pipe.named_steps['impute'].statistics_,np.nanmedian(train[FEATURES],axis=0))
    np.testing.assert_allclose(pipe.named_steps['scale'].mean_,train[FEATURES].mean(axis=0))
    np.testing.assert_allclose(pipe.named_steps['scale'].var_,train[FEATURES].var(axis=0,ddof=0))
    # Separate raw-input linear program: minimize sum |y-Xb|/y.
    # This verifies the objective independently of sklearn's pipeline/weights.
    if meta['candidate']=='mape_linear':
        x=np.column_stack([np.ones(len(train)),train[FEATURES].to_numpy()])
        y=train.cycle_life.to_numpy();n=len(y)
        result=linprog(np.r_[np.zeros(x.shape[1]),1/y],
                       A_ub=np.vstack([np.c_[x,-np.eye(n)],np.c_[-x,-np.eye(n)]]),
                       b_ub=np.r_[y,-y],bounds=[(None,None)]*x.shape[1]+[(0,None)]*n,method='highs')
        assert result.success
        objective=np.sum(np.abs(saved['model'].predict(train[FEATURES])-y)/y)
        np.testing.assert_allclose(objective,result.fun,rtol=1e-8,atol=1e-8)
    prediction=pd.read_csv(HERE/'results/improvement_predictions.csv')
    evaluation=pd.read_csv(HERE/'results/improvement_evaluations.csv')
    previous=joblib.load(HERE/'models/followup_model.joblib')['model']
    ids={'Valid_B1_reused':set(meta['valid_ids']),
         'Test_B2_reused':set(d[(d.batch==2)&d.cycle_life.notna()].cell_id),
         'Additional_B3_reused':set(d[(d.batch==3)&d.cycle_life.notna()].cell_id)}
    for _,r in evaluation.iterrows():
        p=prediction[(prediction.model==r.model)&(prediction.split==r.split)]
        assert set(p.cell_id)==ids[r.split] and len(p)==r.n and p.cell_id.is_unique
        np.testing.assert_allclose(p.actual,d.loc[p.cell_id,'cycle_life'])
        est=saved['model'] if r.model=='Improved' else previous
        np.testing.assert_allclose(p.predicted,est.predict(d.loc[p.cell_id,FEATURES]))
        if r.model=='Improved':
            np.testing.assert_allclose(p.predicted,fitted.predict(d.loc[p.cell_id,FEATURES]))
        assert (p.predicted>0).all()
        error=p.predicted-p.actual
        np.testing.assert_allclose([r.MAPE_pct,r.MAE_cycles,r.RMSE_cycles],
                                  [np.mean(abs(error)/p.actual)*100,np.mean(abs(error)),np.sqrt(np.mean(error**2))])
    fixed=[]
    for fold in json.loads((HERE/'results/cv_folds.json').read_text()):
        a,b=d.loc[fold['train_ids']],d.loc[fold['valid_ids']]
        fixed.append(np.mean(abs(refit(meta['candidate'],a).predict(b[FEATURES])-b.cycle_life)/b.cycle_life)*100)
    np.testing.assert_allclose(fixed,meta['fixed_cv_fold_MAPE_pct'])
    e=evaluation[evaluation.model=='Improved'].set_index('split')
    vm,tm=e.loc['Valid_B1_reused','MAPE_pct'],e.loc['Test_B2_reused','MAPE_pct']
    np.testing.assert_allclose(pd.read_csv(HERE/'results/improvement_performance.csv').iloc[:,1],
                              [np.mean(fixed),vm,tm,vm-np.mean(fixed),tm-vm,tm-9.1])
    missing=pd.read_csv(HERE/'results/improvement_missing_predictions.csv')
    assert set(missing.cell_id)==set(d[d.batch.isin([2,3])&d.cycle_life.isna()].cell_id)
    np.testing.assert_allclose(missing.predicted_cycle_life,saved['model'].predict(d.loc[missing.cell_id,FEATURES]))
    report={'status':'PASS','independently_refitted_candidate_splits':800,
            'nested_selection_train_only':True,'weighted_LAD_optimum_verified':meta['candidate']=='mape_linear',
            'preprocessing_fit_on_train_only':True,'saved_predictions_and_gap_match':True,
            'previous_models_and_predictions_preserved':True,'submitted_day1_unchanged_files':verify_day1()}
    from verify_prediction import verify as verify_prediction
    report['prediction_cli']=verify_prediction()
    (HERE/'results/improvement_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':verify()
