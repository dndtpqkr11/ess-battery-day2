"""Audit follow-up selection, repeated splits, saved model and reused scores."""
from pathlib import Path
import json,hashlib
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression,Ridge
from train import make,verify_day1

HERE=Path(__file__).resolve().parent

def verify():
    d=pd.read_csv(HERE/'data/features.csv').set_index('cell_id',drop=False)
    protocol=json.loads((HERE/'results/followup_protocol.json').read_text())
    audit=json.loads((HERE/'results/followup_validation.json').read_text())
    for filename,digest in audit['primary_artifacts_sha256'].items():
        assert hashlib.sha256((HERE/filename).read_bytes()).hexdigest()==digest,filename
    original=json.loads((HERE/'results/selected_model.json').read_text())
    assert protocol['train_ids']==original['train_ids']
    assert set(protocol['train_ids']).isdisjoint(protocol['excluded_original_holdout_ids'])
    splits=json.loads((HERE/'results/policy_stability_splits.json').read_text())
    scores=pd.read_csv(HERE/'results/policy_stability_scores.csv')
    assert len(splits)==20 and len(scores)==140
    for fold in splits:
        a,b=d.loc[fold['train_ids']],d.loc[fold['valid_ids']]
        assert set(a.cell_id)|set(b.cell_id)==set(original['train_ids'])
        assert set(a.cell_id).isdisjoint(b.cell_id) and set(a.policy).isdisjoint(b.policy)
        for name,fs in protocol['candidates'].items():
            est=make(LinearRegression(),False) if name=='delta_linear_raw' else make(Ridge(alpha=1),True)
            est.fit(a[fs],a.cycle_life);prediction=est.predict(b[fs])
            value=float(np.mean(np.abs(prediction-b.cycle_life.to_numpy())/b.cycle_life.to_numpy())*100)
            stored=scores[(scores.seed==fold['seed'])&(scores.candidate==name)].MAPE_pct.item()
            np.testing.assert_allclose(value,stored,rtol=1e-10)
    choice=json.loads((HERE/'results/followup_selected.json').read_text())
    means=scores.groupby('candidate').MAPE_pct.mean()
    assert choice['candidate']==means.idxmin()
    np.testing.assert_allclose(choice['mean_repeated_holdout_MAPE_pct'],means.min())
    saved=joblib.load(HERE/'models/followup_model.joblib');fs=saved['features']
    assert saved['selection']==choice and fs==protocol['candidates'][choice['candidate']]
    train=d.loc[original['train_ids']];fit=clone(saved['model']).fit(train[fs],train.cycle_life)
    pred=pd.read_csv(HERE/'results/followup_predictions.csv');evaluation=pd.read_csv(HERE/'results/followup_evaluations.csv')
    ids={'Valid_B1_reused':set(original['valid_ids']),
         'Test_B2_reused':set(d[(d.batch==2)&d.cycle_life.notna()].cell_id),
         'Additional_B3_reused':set(d[(d.batch==3)&d.cycle_life.notna()].cell_id)}
    for _,row in evaluation.iterrows():
        p=pred[pred.split==row.split];assert set(p.cell_id)==ids[row.split]
        np.testing.assert_allclose(p.actual,d.loc[p.cell_id,'cycle_life'])
        np.testing.assert_allclose(p.predicted,saved['model'].predict(d.loc[p.cell_id,fs]),rtol=1e-10)
        np.testing.assert_allclose(p.predicted,fit.predict(d.loc[p.cell_id,fs]),rtol=1e-10)
        np.testing.assert_allclose(row.MAPE_pct,np.mean(np.abs(p.predicted-p.actual)/p.actual)*100)
        np.testing.assert_allclose(row.MAE_cycles,np.mean(np.abs(p.predicted-p.actual)))
        np.testing.assert_allclose(row.RMSE_cycles,np.sqrt(np.mean((p.predicted-p.actual)**2)))
    cv=[]
    for fold in json.loads((HERE/'results/cv_folds.json').read_text()):
        a,b=d.loc[fold['train_ids']],d.loc[fold['valid_ids']]
        est=clone(saved['model']).fit(a[fs],a.cycle_life)
        cv.append(np.mean(np.abs(est.predict(b[fs])-b.cycle_life)/b.cycle_life)*100)
    np.testing.assert_allclose(cv,choice['fixed_cv_fold_MAPE_pct'],rtol=1e-10)
    perf=pd.read_csv(HERE/'results/followup_performance.csv').iloc[:,1].to_numpy()
    vm=evaluation[evaluation.split=='Valid_B1_reused'].MAPE_pct.item();tm=evaluation[evaluation.split=='Test_B2_reused'].MAPE_pct.item()
    np.testing.assert_allclose(perf,[np.mean(cv),vm,tm,vm-np.mean(cv),tm-vm,tm-9.1])
    missing=pd.read_csv(HERE/'results/followup_missing_target_predictions.csv')
    assert set(missing.cell_id)==set(d[d.batch.isin([2,3])&d.cycle_life.isna()].cell_id)
    np.testing.assert_allclose(missing.predicted_cycle_life,saved['model'].predict(d.loc[missing.cell_id,fs]))
    identity=pd.read_csv(HERE/'results/cell_identity.csv');assert len(identity)==139 and identity.barcode.is_unique and set(identity.cell_id)==set(d.cell_id)
    if (HERE/'results/B3_quality_cohort.csv').exists():
        cohort=pd.read_csv(HERE/'results/B3_quality_cohort.csv')
        assert len(cohort)==46 and cohort.retained.sum()==40
        assert set(cohort[~cohort.retained].cell_id)=={'b3c37','b3c23','b3c32','b3c2','b3c42','b3c43'}
        sensitivity=pd.read_csv(HERE/'results/B3_quality_sensitivity.csv')
        g=d.loc[cohort[cohort.retained].cell_id]
        for _,r in sensitivity.iterrows():
            m=saved if r.model=='followup_Linear' else joblib.load(HERE/'models/selected_model.joblib')
            predicted=m['model'].predict(g[m['features']])
            np.testing.assert_allclose(r.MAPE_pct,np.mean(np.abs(predicted-g.cycle_life)/g.cycle_life)*100)
    report={'status':'PASS','independently_refitted_candidate_splits':140,'followup_selection_uses_B1_train_only':True,
            'saved_followup_predictions_and_gap_match':True,'primary_model_and_scores_preserved':True,
            'unique_normalized_barcodes':139,'submitted_day1_unchanged_files':verify_day1()}
    (HERE/'results/followup_independent_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':verify()
