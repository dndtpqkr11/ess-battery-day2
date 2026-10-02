"""Third-round train-only comparison of robust and MAPE-aligned regression.

Previous artifacts remain unchanged. Nested group CV measures the complete
candidate selection procedure; reused external scores never select the model.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from train import make, metrics, verify_day1

HERE = Path(__file__).resolve().parent
FEATURES = ['log_dq_var']


def specs():
    # Fixed before external evaluation. No extra feature search on 28 cells.
    return {
        'previous_linear': (make(LinearRegression(), False), None),
        'relative_squared_linear': (make(LinearRegression(), False), 2),
        'mape_linear': (make(QuantileRegressor(quantile=.5, alpha=0, solver='highs'), False), 1),
        'absolute_linear': (make(QuantileRegressor(quantile=.5, alpha=0, solver='highs'), False), None),
        'huber_raw': (make(HuberRegressor(epsilon=1.35, max_iter=2000), False), None),
        'huber_log': (make(HuberRegressor(epsilon=1.35, max_iter=2000), True), None),
        'median_log': (make(QuantileRegressor(quantile=.5, alpha=0, solver='highs'), True), None),
        'linear_log': (make(LinearRegression(), True), None),
        'ridge_raw_1': (make(Ridge(alpha=1), False), None),
        'ridge_raw_10': (make(Ridge(alpha=10), False), None),
    }


def fit(name, data):
    est, power = specs()[name]
    kwargs = {}
    if power:
        weights = data.cycle_life.to_numpy(dtype=float) ** (-power)
        # Weights use labels of this training fold only.
        kwargs['model__sample_weight'] = weights / weights.mean()
    return clone(est).fit(data[FEATURES], data.cycle_life, **kwargs)


def select(data, seeds=range(20)):
    rows, splits = [], []
    for seed in seeds:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
                    .split(data, groups=data.policy))
        t, v = data.iloc[a], data.iloc[b]
        assert set(t.policy).isdisjoint(v.policy)
        splits.append({'seed': seed, 'train_ids': t.cell_id.tolist(), 'valid_ids': v.cell_id.tolist()})
        for name in specs():
            p = fit(name, t).predict(v[FEATURES])
            rows.append({'candidate': name, 'seed': seed,
                         **metrics(v.cycle_life.to_numpy(), p)})
    scores = pd.DataFrame(rows)
    summary = scores.groupby('candidate').MAPE_pct.agg(
        mean_MAPE_pct='mean', std_MAPE_pct='std', max_MAPE_pct='max').reset_index()
    summary = summary.sort_values(['mean_MAPE_pct', 'candidate']).reset_index(drop=True)
    return summary.iloc[0].candidate, scores, summary, splits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    verify_day1()
    protected = [HERE/'models/selected_model.joblib', HERE/'models/followup_model.joblib',
                 HERE/'results/predictions.csv', HERE/'results/followup_predictions.csv']
    before = {str(p.relative_to(HERE)): digest(p) for p in protected}
    d = pd.read_csv(HERE/'data/features.csv').set_index('cell_id', drop=False)
    old = joblib.load(HERE/'models/followup_model.joblib')
    primary = joblib.load(HERE/'models/selected_model.joblib')['selection']
    train = d.loc[primary['train_ids']].reset_index(drop=True)
    protocol = {
        'train_ids': train.cell_id.tolist(), 'excluded_holdout_ids': primary['valid_ids'],
        'features': FEATURES, 'seeds': list(range(20)), 'candidates': list(specs()),
        'selection_rule': 'Minimum mean MAPE on 20 fixed B1 training policy hold-outs; tie by candidate name',
        'weight_rule': 'MAPE LAD: normalized 1/y; relative squared: normalized 1/y^2; fit-fold labels only',
        'nested_rule': '3 outer GroupKFold folds; each repeats the full 20-split candidate selection on outer train only',
        'interpretation': 'All data have been explored previously. Nested CV is internal diagnostic, not a fresh external test.',
        'input_sha256': digest(HERE/'data/features.csv'), 'protected_sha256': before,
    }
    (HERE/'results/improvement_protocol.json').write_text(json.dumps(protocol, indent=2))
    name, scores, summary, splits = select(train)
    scores.to_csv(HERE/'results/improvement_scores.csv', index=False)
    summary.to_csv(HERE/'results/improvement_summary.csv', index=False)
    (HERE/'results/improvement_splits.json').write_text(json.dumps(splits, indent=2))
    print(summary.to_string(index=False), flush=True)

    nested, nested_predictions, nested_search = [], [], []
    fixed = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=3).split(train, groups=train.policy), 1):
        t, v = train.iloc[a], train.iloc[b]
        selected, inner_scores, _, _ = select(t)
        inner_scores['outer_fold'] = fold
        nested_search.append(inner_scores)
        p = fit(selected, t).predict(v[FEATURES])
        baseline_p = fit('previous_linear', t).predict(v[FEATURES])
        nested.append({'fold': fold, 'selected': selected, 'train_ids': t.cell_id.tolist(),
                       'valid_ids': v.cell_id.tolist(), **metrics(v.cycle_life.to_numpy(), p),
                       'previous_MAPE_pct': metrics(v.cycle_life.to_numpy(), baseline_p)['MAPE_pct']})
        nested_predictions.extend({'fold': fold, 'cell_id': cid, 'actual': y, 'predicted': q,
                                   'previous_predicted': prev}
                                  for cid, y, q, prev in zip(v.cell_id, v.cycle_life, p, baseline_p))
        fixed.append(metrics(v.cycle_life.to_numpy(), fit(name, t).predict(v[FEATURES]))['MAPE_pct'])
        print(f'Nested fold {fold}: {selected}, MAPE={nested[-1]["MAPE_pct"]:.2f}%', flush=True)
    pd.concat(nested_search).to_csv(HERE/'results/improvement_nested_search.csv', index=False)
    (HERE/'results/improvement_nested_folds.json').write_text(json.dumps(nested, indent=2))
    pd.DataFrame(nested_predictions).to_csv(HERE/'results/improvement_nested_predictions.csv', index=False)
    selected = {
        'candidate': name, 'features': FEATURES, 'train_ids': train.cell_id.tolist(),
        'valid_ids': primary['valid_ids'], 'input_sha256': protocol['input_sha256'],
        'selection_rule': protocol['selection_rule'],
        'mean_repeated_holdout_MAPE_pct': float(summary.iloc[0].mean_MAPE_pct),
        'fixed_cv_MAPE_pct': float(np.mean(fixed)), 'fixed_cv_std_pct': float(np.std(fixed)),
        'fixed_cv_fold_MAPE_pct': fixed,
        'nested_cv_MAPE_pct': float(np.mean([r['MAPE_pct'] for r in nested])),
        'nested_cv_std_pct': float(np.std([r['MAPE_pct'] for r in nested])),
        'nested_previous_MAPE_pct': float(np.mean([r['previous_MAPE_pct'] for r in nested])),
        'feature_ranges': {f: {'min': float(train[f].min()), 'max': float(train[f].max())} for f in FEATURES},
        'training_target_range': [float(train.cycle_life.min()), float(train.cycle_life.max())],
    }
    (HERE/'results/improvement_selected.json').write_text(json.dumps(selected, indent=2))
    model = fit(name, train)
    joblib.dump({'model': model, 'features': FEATURES, 'selection': selected}, HERE/'models/improved_model.joblib')
    # Selection is now frozen. Only now evaluate reused hold-out/external labels.
    evaluations, predictions = [], []
    for split, g in [('Valid_B1_reused', d.loc[primary['valid_ids']]),
                     ('Test_B2_reused', d[(d.batch==2)&d.cycle_life.notna()]),
                     ('Additional_B3_reused', d[(d.batch==3)&d.cycle_life.notna()])]:
        for label, estimator in [('Improved', model), ('Previous', old['model'])]:
            p = estimator.predict(g[FEATURES])
            evaluations.append({'model': label, 'split': split, **metrics(g.cycle_life.to_numpy(), p)})
            predictions.extend({'model': label, 'split': split, 'cell_id': cid, 'actual': y,
                                'predicted': q, 'APE_pct': abs(y-q)/y*100}
                               for cid, y, q in zip(g.cell_id, g.cycle_life, p))
    evaluation = pd.DataFrame(evaluations)
    evaluation.to_csv(HERE/'results/improvement_evaluations.csv', index=False)
    pd.DataFrame(predictions).to_csv(HERE/'results/improvement_predictions.csv', index=False)
    missing = d[d.batch.isin([2,3])&d.cycle_life.isna()].copy()
    missing['predicted_cycle_life'] = model.predict(missing[FEATURES])
    missing[['cell_id','batch','predicted_cycle_life']].to_csv(HERE/'results/improvement_missing_predictions.csv', index=False)
    e = evaluation[evaluation.model=='Improved'].set_index('split')
    vm, tm = float(e.loc['Valid_B1_reused','MAPE_pct']), float(e.loc['Test_B2_reused','MAPE_pct'])
    pd.DataFrame([
        ['Train (Batch 1 CV)', selected['fixed_cv_MAPE_pct'], 'Fixed 3-fold diagnostic after selection'],
        ['Valid (Batch 1 Hold-out)', vm, 'Previously seen; excluded from selection'],
        ['Test (Batch 2)', tm, 'Previously seen; reused evaluation'],
        ['Gap (Train-Valid)', vm-selected['fixed_cv_MAPE_pct'], 'Valid-Train; percentage points'],
        ['Gap (Valid-Test)', tm-vm, 'Test-Valid; percentage points'],
        ['Gap (Target-Test)', tm-9.1, 'Test-9.1; percentage points'],
    ], columns=['구분','MAPE (%) / Gap (%p)','비고']).to_csv(HERE/'results/improvement_performance.csv', index=False)
    assert all(digest(HERE/p)==h for p,h in before.items())
    print(evaluation.to_string(index=False), flush=True)
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == '__main__':
    run()
