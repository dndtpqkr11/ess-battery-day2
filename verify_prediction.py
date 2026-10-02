"""Exercise the public CLI in a separate process with target-free inputs."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent


def verify():
    artifact=joblib.load(HERE/'models/improved_model.joblib')
    d=pd.read_csv(HERE/'data/features.csv')
    early=d[['cell_id',*artifact['features']]].copy()
    with tempfile.TemporaryDirectory(prefix='day2-cli-') as tmp:
        root=Path(tmp);input_path=root/'early.csv';output_path=root/'prediction.csv'
        early.to_csv(input_path,index=False)
        cmd=[sys.executable,str(HERE/'predict.py'),'--input',str(input_path),'--output',str(output_path)]
        subprocess.run([*cmd,'--diagnostics'],check=True,capture_output=True,text=True)
        output=pd.read_csv(output_path)
        assert output.cell_id.tolist()==early.cell_id.tolist()
        p=artifact['model'].predict(early[artifact['features']])
        np.testing.assert_allclose(output.predicted_cycle_life,p)
        flags=[]
        for f in artifact['features']:
            lo,hi=early.loc[d.cell_id.isin(artifact['selection']['train_ids']),f].agg(['min','max'])
            expected=(early[f]<lo)|(early[f]>hi)
            np.testing.assert_array_equal(output[f'{f}_outside_train_range'],expected)
            flags.append(expected.to_numpy())
        np.testing.assert_array_equal(output.any_feature_outside_train_range,np.any(flags,axis=0))
        np.testing.assert_array_equal(output.missing_feature_count,early[artifact['features']].isna().sum(axis=1))
        target=d.loc[d.cell_id.isin(artifact['selection']['train_ids']),'cycle_life']
        np.testing.assert_array_equal(output.prediction_outside_train_target_range,(p<target.min())|(p>target.max()))
        baseline=joblib.load(HERE/'models/followup_model.joblib')
        subprocess.run([*cmd,'--model',str(HERE/'models/followup_model.joblib')],check=True,capture_output=True,text=True)
        np.testing.assert_allclose(pd.read_csv(output_path).predicted_cycle_life,baseline['model'].predict(early[baseline['features']]))
        early[['cell_id']].to_csv(input_path,index=False)
        bad=subprocess.run(cmd,capture_output=True,text=True)
        assert bad.returncode!=0 and 'Missing required features' in bad.stderr
    report={'status':'PASS','target_free_rows':len(early),'default_is_improved_model':True,
            'range_diagnostics_verified':True,'previous_model_supported':True,'missing_feature_rejected':True}
    (HERE/'results/improvement_cli_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':verify()
