"""Independently audit saved splits, fitted preprocessing, CV and predictions."""
from pathlib import Path
import argparse,json,hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

HERE=Path(__file__).resolve().parent

def verify(raw_dir=None):
    d=pd.read_csv(HERE/'data/features.csv').set_index('cell_id',drop=False)
    split=pd.read_csv(HERE/'results/split.csv')
    artifact=joblib.load(HERE/'models/selected_model.joblib');meta=artifact['selection'];fs=artifact['features']
    allowed={'log_dq_var','QD_slope_100','chargetime_mean_100','Tavg_mean_100','IR_mean_100','charge_I_time_std_5'}
    assert set(fs)<=allowed and len(set(fs))==len(fs)
    assert d.index.is_unique and split.cell_id.is_unique
    assert hashlib.sha256((HERE/'data/features.csv').read_bytes()).hexdigest()==meta['input_sha256']
    train=d.loc[split[split.split=='train'].cell_id];valid=d.loc[split[split.split=='valid'].cell_id]
    assert len(train)==28 and len(valid)==8
    eligible=d[(d.batch==1)&~d.possible_censored&d.cycle_life.notna()]
    assert set(split.cell_id)==set(eligible.cell_id)
    assert set(train.policy).isdisjoint(valid.policy)
    assert set(train.cell_id)==set(meta['train_ids']) and set(valid.cell_id)==set(meta['valid_ids'])
    model=artifact['model'];pipe=model.regressor_
    medians=np.nanmedian(train[fs].to_numpy(),axis=0)
    np.testing.assert_allclose(pipe.named_steps['impute'].statistics_,medians)
    imputed=train[fs].fillna(dict(zip(fs,medians))).to_numpy()
    np.testing.assert_allclose(pipe.named_steps['scale'].mean_,imputed.mean(axis=0))
    np.testing.assert_allclose(pipe.named_steps['scale'].var_,imputed.var(axis=0))
    folds=json.loads((HERE/'results/cv_folds.json').read_text());scores=[];cv_ids=[]
    for fold in folds:
        a=d.loc[fold['train_ids']];b=d.loc[fold['valid_ids']]
        assert set(a.cell_id)|set(b.cell_id)==set(train.cell_id)
        assert set(a.cell_id).isdisjoint(b.cell_id) and set(a.policy).isdisjoint(b.policy)
        assert set(a.cell_id).isdisjoint(valid.cell_id) and set(b.cell_id).isdisjoint(valid.cell_id)
        est=clone(model).fit(a[fs],a.cycle_life)
        prediction=est.predict(b[fs]);scores.append(np.mean(np.abs(b.cycle_life-prediction)/b.cycle_life)*100)
        cv_ids.extend(b.cell_id.tolist())
    assert len(cv_ids)==len(set(cv_ids))==len(train)
    np.testing.assert_allclose(np.mean(scores),meta['cv_MAPE_pct'],rtol=1e-10)
    np.testing.assert_allclose(np.std(scores),meta['cv_std_pct'],rtol=1e-10)
    candidates=pd.read_csv(HERE/'results/candidates.csv')
    np.testing.assert_allclose(meta['cv_MAPE_pct'],candidates[candidates.model!='DummyMedian'].cv_MAPE_pct.min())
    pred=pd.read_csv(HERE/'results/predictions.csv');evaluation=pd.read_csv(HERE/'results/evaluations.csv')
    split_ids={'Valid_B1':set(valid.cell_id),'Test_B2':set(d[(d.batch==2)&d.cycle_life.notna()].cell_id),
               'Additional_B3':set(d[(d.batch==3)&d.cycle_life.notna()].cell_id)}
    for _,row in evaluation.iterrows():
        p=pred[(pred.model==row.model)&(pred.split==row.split)]
        assert set(p.cell_id)==split_ids[row.split] and len(p)==int(row.n)
        np.testing.assert_allclose(p.actual,d.loc[p.cell_id,'cycle_life'])
        expected=model.predict(d.loc[p.cell_id,fs]) if row.model=='Selected' else np.full(len(p),train.cycle_life.median())
        np.testing.assert_allclose(p.predicted,expected,rtol=1e-10)
        error=p.predicted-p.actual
        np.testing.assert_allclose(row.MAPE_pct,np.mean(np.abs(error)/p.actual)*100)
        np.testing.assert_allclose(row.MAE_cycles,np.mean(np.abs(error)))
        np.testing.assert_allclose(row.RMSE_cycles,np.sqrt(np.mean(error**2)))
        assert (p.predicted>0).all()
    missing=pd.read_csv(HERE/'results/missing_target_predictions.csv')
    assert set(missing.cell_id)==set(d[(d.batch.isin([2,3]))&d.cycle_life.isna()].cell_id)
    np.testing.assert_allclose(missing.predicted_cycle_life,model.predict(d.loc[missing.cell_id,fs]))
    p=pd.read_csv(HERE/'results/performance_reporting.csv').iloc[0]
    np.testing.assert_allclose([p.Gap_Train_Valid_pp,p.Gap_Valid_Test_pp,p.Gap_Target_Test_pp],
        [p.Valid_B1_MAPE_pct-p.Train_B1_CV_MAPE_pct,p.Test_B2_MAPE_pct-p.Valid_B1_MAPE_pct,p.Test_B2_MAPE_pct-9.1])
    raw_verified=False
    if raw_dir:
        from extract_features import extract
        extracted=extract(raw_dir).set_index('cell_id').sort_index()
        pd.testing.assert_frame_equal(d.drop(columns='cell_id').sort_index()[extracted.columns],extracted,
                                      check_exact=False,rtol=1e-10,atol=1e-12)
        raw_verified=True
    from train import verify_day1
    protected=verify_day1()
    report={'status':'PASS','train_valid_policy_disjoint':True,'cv_policy_disjoint':True,
            'preprocessing_fit_on_train_only':True,'cv_independently_refitted':True,
            'saved_predictions_and_metrics_match':True,'raw_features_verified':raw_verified,
            'scored_selected_predictions':sum(len(ids) for ids in split_ids.values()),
            'unscored_missing_targets':len(missing),'submitted_day1_unchanged_files':protected,
            'input_sha256':meta['input_sha256']}
    if (HERE/'models/followup_model.joblib').exists():
        from verify_followup import verify as verify_followup
        report['followup']=verify_followup()
    if (HERE/'models/improved_model.joblib').exists():
        from verify_improvement import verify as verify_improvement
        report['improvement']=verify_improvement()
    (HERE/'results/final_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw-dir',type=Path)
    verify(parser.parse_args().raw_dir)
