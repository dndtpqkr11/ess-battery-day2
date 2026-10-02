from pathlib import Path
import os, json
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mini-project-day2-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def run():
    d=pd.read_csv(HERE/'data/features.csv'); split=pd.read_csv(HERE/'results/split.csv')
    train=d[d.cell_id.isin(split[split.split=='train'].cell_id)]
    valid=d[d.cell_id.isin(split[split.split=='valid'].cell_id)]
    selected=json.loads((HERE/'results/selected_model.json').read_text())
    p=pd.read_csv(HERE/'results/performance_reporting.csv').iloc[0]
    pd.DataFrame([
        ['Train (Batch 1 CV)',p.Train_B1_CV_MAPE_pct,'3-fold mean; std='+str(p.Train_B1_CV_std_pct)],
        ['Valid (Batch 1 Hold-out)',p.Valid_B1_MAPE_pct,'8 cells; policy group split'],
        ['Test (Batch 2)',p.Test_B2_MAPE_pct,'39 valid targets'],
        ['Gap (Train-Valid)',p.Gap_Train_Valid_pp,'Valid-Train; percentage points'],
        ['Gap (Valid-Test)',p.Gap_Valid_Test_pp,'Test-Valid; percentage points'],
        ['Gap (Target-Test)',p.Gap_Target_Test_pp,'Test-9.1; percentage points']
    ],columns=['구분','MAPE (%) / Gap (%p)','비고']).to_csv(HERE/'results/model_performance.csv',index=False)
    ev=pd.read_csv(HERE/'results/evaluations.csv'); candidates=pd.read_csv(HERE/'results/candidates.csv')
    pred=pd.read_csv(HERE/'results/predictions.csv')
    ranges=[]; support=[]
    for batch in [2,3]:
        g=d[(d.batch==batch)&d.cycle_life.notna()]
        support.append({'batch':batch,'n':len(g),'train_life_min':train.cycle_life.min(),'train_life_max':train.cycle_life.max(),
                        'life_below_train_min':int((g.cycle_life<train.cycle_life.min()).sum()),'life_above_train_max':int((g.cycle_life>train.cycle_life.max()).sum()),
                        'unseen_policy_cells':int((~g.policy.isin(train.policy)).sum())})
        for f in selected['features']:
            lo,hi=train[f].min(),train[f].max(); z=g[f]
            ranges.append({'batch':batch,'feature':f,'train_min':lo,'train_max':hi,'test_min':z.min(),'test_max':z.max(),'missing_n':int(z.isna().sum()),'outside_train_range_n':int(((z<lo)|(z>hi)).sum()),'observed_n':int(z.notna().sum())})
    pd.DataFrame(ranges).to_csv(HERE/'results/feature_shift.csv',index=False)
    support=pd.DataFrame(support); support.to_csv(HERE/'results/target_support.csv',index=False)
    test=pred[(pred.model=='Selected')&(pred.split=='Test_B2')].copy()
    test['signed_error']=test.predicted-test.actual
    groups=[]
    for name,mask in [('short_lt500',test.actual<500),('mid_500_1000',test.actual.between(500,1000)),('long_gt1000',test.actual>1000)]:
        a=test[mask]
        groups.append({'group':name,'n':len(a),'MAPE_pct':a.APE_pct.mean(),'mean_signed_error_cycles':a.signed_error.mean()})
    pd.DataFrame(groups).to_csv(HERE/'results/test_error_by_life.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(test.actual,test.signed_error,color='#f59e0b'); ax.axhline(0,color='gray',ls='--'); ax.set(xlabel='Actual cycle life (Batch 2)',ylabel='Predicted - actual (cycles)',title='External batch prediction error'); fig.tight_layout(); fig.savefig(HERE/'figures/batch2_error.png',dpi=170); plt.close(fig)
    b2=support[support.batch==2].iloc[0]
    dummy=ev[(ev.model=='DummyMedian')&(ev.split=='Test_B2')].iloc[0]
    gap_to_linear=selected['cv_MAPE_pct']-candidates[(candidates.model=='Linear')&(candidates.feature_set=='delta_only')&(candidates.target=='raw')].cv_MAPE_pct.iloc[0]
    report=f'''# ESS 배터리 수명 예측 — DAY 2

울산 4반 박세웅 (U115) | 회귀: 초기 100사이클로 총 수명 `cycle_life` 예측

학습 충전 정책 그룹 CV에서 **{selected['model']} / {selected['feature_set']} / {selected['target']} 타깃**을 선택했다. Batch 2 MAPE는 **{p.Test_B2_MAPE_pct:.2f}%**, 논문 목표 9.1% 대비 **+{p.Gap_Target_Test_pp:.2f}%p**다. 내부 검증 오차는 작았으나 외부 배치에서 일반화에 실패했다. 예비 실험의 후보·분할을 고정해 다시 실행한 결과이며, 이번 작업에서 테스트 결과에 맞춘 모델 교체는 하지 않았다.

## EDA → 구현

| DAY 1 발견 | DAY 2 반영 |
| --- | --- |
| B1의 <550사이클 셀은 1/46 | 분류 대신 총 사이클 수 회귀 |
| 수명 중앙값 B1/B2/B3 = 858.5/472/1005.5 | B1 학습, B2 필수 외부 평가, B3 추가 평가 |
| 말기 용량 감소 가속과 knee 후보 | 전체 열화 곡선·knee를 예측 입력에서 제외 |
| log ΔQ 분산–수명 Spearman = -0.871/-0.709/-0.797 | Q100(V)-Q10(V)의 log 분산을 핵심 특징으로 사용 |
| 충전 첫 C-rate의 관계는 배치별로 다름 | 정책 자체는 예측 변수 대신 분할 그룹으로 사용 |
| ΔQ 특징 간 중복 0.971–0.989, 전류 표준편차와 log ΔQ 중복 0.906 | 단일 특징 기준과 소수 특징 정규화 모델 비교 |
| B1 측정 종료 경고 10셀, B2 전체 IR=0인 유효 수명 6셀 | 경고 셀의 학습 제외, IR 0은 결측으로 대체 |

선택된 6개 특징: `{', '.join(selected['features'])}`. ΔQ는 실제 cycle 번호 10/100, 용량 기울기·충전 시간·온도·IR은 cycle 2–100, 충전 전류 표준편차는 cycle 2–5에서 계산한다. 특징 정의와 원본 재추출은 [data/README.md](data/README.md)에 정리했다.

## 모델·파이프라인

학습 중앙값 결측 대체 → 표준화 → 회귀 모델. 각 CV fold에서 전처리를 다시 fit한다. log10 타깃은 역변환한 사이클 수로 MAPE를 계산한다. 미래 열화 속도, 전체 관측 길이, 종료 용량, knee, 배치 번호, 셀 ID, 정책 문자열은 입력 허용 목록에 없다.

DummyMedian, Linear, Ridge, ElasticNet, 얕은 RandomForest를 비교했다. 단일 ΔQ/compact/compact_charge와 원본/log10 타깃을 합쳐 85개 설정을 학습 그룹 CV로 비교했다. 작은 표본의 특징 중복 때문에 Ridge/ElasticNet을, 비선형성 비교를 위해 깊이 2–3의 RandomForest를 사용했다.

선택 기준은 **학습 그룹 CV 평균 MAPE 최소값**이다. 최종 Ridge 파라미터는 `{selected['params']}`. 단일 ΔQ Linear보다 CV 오차가 {abs(gap_to_linear):.3f}%p 작을 뿐이므로 우월성을 확정하지 않는다. 상세 후보 비교는 [results/candidates.csv](results/candidates.csv)에 있다.

## 데이터 분할

B1 46셀에서 종료 경고 10셀을 제외한 36셀을 정책 그룹 hold-out(seed=42)으로 나눴다. 학습 **{len(train)}셀/{train.policy.nunique()}정책**, 검증 **{len(valid)}셀/{valid.policy.nunique()}정책**. 학습 부분에 GroupKFold 3-fold를 적용했다. 셀과 정책은 train/valid 및 CV 양쪽에 겹치지 않는다. B2/B3는 학습·전처리 fit·하이퍼파라미터 선택에 사용하지 않는다.

Train CV는 hold-out을 제외한 28셀의 결과이며 모델 선택에도 사용됐다. 최종 독립 성능 추정치로 해석하지 않는다. 전체 36셀 재학습 점수와 섞지 않고 동일한 28셀 학습 모델로 평가한다.

## 성능 결과

MAPE 단위는 %, Gap 단위는 %p다. 노션 예시의 양수=오차 증가 해석에 맞춰 Gap을 `Valid−Train`, `Test−Valid`, `Test−Target`으로 정의했다. 항목명은 노션 형식을 유지한다.

| 구분 | MAPE (%) / Gap (%p) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | {p.Train_B1_CV_MAPE_pct:.2f} ± {p.Train_B1_CV_std_pct:.2f} | 3-fold 평균 ± 표준편차; 신뢰구간 아님 |
| Valid (Batch 1 Hold-out) | {p.Valid_B1_MAPE_pct:.2f} | 8셀, 정책 그룹 분리 |
| Test (Batch 2) | {p.Test_B2_MAPE_pct:.2f} | 유효 수명 39셀 |
| Gap (Train-Valid) | {p.Gap_Train_Valid_pp:+.2f} | Valid−Train |
| Gap (Valid-Test) | {p.Gap_Valid_Test_pp:+.2f} | Test−Valid; 배치 일반화 오차 증가 |
| Gap (Target-Test) | {p.Gap_Target_Test_pp:+.2f} | Test−9.1; 논문 목표 미달 |

| 평가 집합 | n | 선택 모델 MAPE | 중앙값 기준 MAPE | 선택 모델 MAE(사이클) |
| --- | ---: | ---: | ---: | ---: |
'''
    for label in ['Valid_B1','Test_B2','Additional_B3']:
        a=ev[(ev.model=='Selected')&(ev.split==label)].iloc[0]; b=ev[(ev.model=='DummyMedian')&(ev.split==label)].iloc[0]
        report+=f'| {label} | {int(a.n)} | {a.MAPE_pct:.2f}% | {b.MAPE_pct:.2f}% | {a.MAE_cycles:.1f} |\n'
    report+=f'''
B2 수명 결측 8셀, B3 결측 2셀은 정답을 대체하지 않고 평가에서 제외했다. 예측만 별도 저장했다. B2에서는 선택 모델이 중앙값 기준 모델보다도 나빴다.

## 오류 분석과 ESS 해석

학습 수명 범위는 {train.cycle_life.min():.0f}–{train.cycle_life.max():.0f}사이클이다. B2 39셀 중 {int(b2.life_below_train_min)}셀은 학습 최솟값보다 짧고, {int(b2.life_above_train_max)}셀은 최댓값보다 길다. {int(b2.unseen_policy_cells)}셀은 학습에서 보지 않은 정책이다. 짧은 수명 셀을 과대 예측하며, 실제 {test.loc[test.APE_pct.idxmax(),'actual']:.0f}사이클인 `{test.loc[test.APE_pct.idxmax(),'cell_id']}`을 {test.loc[test.APE_pct.idxmax(),'predicted']:.0f}사이클로 예측했다. 특징 이동·작은 표본·중복 특징은 가능한 원인이며 인과적 설명은 아니다.

초기 수명 예측은 BESS 셀 선별과 교체 계획의 후보 정보로 활용할 수 있다. 그러나 이 모델의 단수명 과대 예측을 교체 결정에 쓰면 교체를 늦출 수 있다. 현재 결과로 실제 운영 효과를 주장할 수 없다. 현장 적용에는 해당 셀 화학계·온도·부하에서의 별도 검증, 예측 불확실성, 안전 여유와 SOH 기반 감시가 필요하다.

후속 개선은 B1 내부에서 중복 특징 축소와 정책별 안정성을 확인하고, 단수명 학습 표본을 확보하는 방향이다. 이미 본 B2 점수로 재튜닝한다면 재사용 테스트임을 명시하고 새 외부 평가 자료가 필요하다. 세 배치 EDA와 예비 테스트를 이미 수행했으므로 완전히 보지 않은 테스트는 아니다. 논문과 파일 버전·분할이 다르고 교차 배치 동일 셀 여부가 미확정이므로 논문 성능의 정확한 재현으로 해석하지 않는다.

![예측과 실제 수명](figures/prediction_scatter.png)
![Batch 2 오류](figures/batch2_error.png)
![학습 CV 후보 비교](figures/cv_comparison.png)

## 실행

이 폴더 자체를 저장소 루트로 사용한다. 제공된 `data/features.csv`로 원본 MAT 없이 학습·평가를 재현할 수 있다. Python 3.11 기준이다.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/python summarize.py
.venv/bin/python verify.py
.venv/bin/python predict.py --input data/features.csv --output tmp/predictions.csv
```

원본에서 특징을 다시 생성하는 방법은 [data/README.md](data/README.md)에 있다. 기존 프로젝트에서 실행할 때는 상위 `.venv/bin/python day2/train.py`를 사용할 수 있다. 제출된 DAY 1의 40개 파일 해시는 실행 전후에 확인하며 수정하지 않는다. 독립 저장소에서는 상위 DAY 1 파일이 필요 없다.

## 파일과 검증

- `extract_features.py`, `train.py`, `predict.py`, `verify.py`: 원본 특징 추출·학습·추론·독립 검증.
- `summarize.py`: 성능 표·실패 진단·README 생성.
- `data/features.csv`, `models/selected_model.joblib`: 재현 입력과 저장 모델.
- `results/`: 후보·fold별 CV·분할·선택 근거·셀별 예측·성능 표·분포 이동·검증 기록.
- [VALIDATION.md](VALIDATION.md): 노션 DAY 2 평가 항목 대응과 실행 검증.

공개 GitHub 링크가 DAY 2 제출물이다. 이 폴더는 공개 저장소용 자료이며, GitHub 업로드와 Slack 제출은 별도로 수행한다.

## 출처와 구성

[과제](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb), [Kaggle 데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle), [논문 공개 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation). Severson et al. (2019), *Nature Energy* 4, 383–391, DOI: 10.1038/s41560-019-0356-8.

박세웅: EDA, 특징 설계, 모델 개발, 성능 평가 및 결과 검토.
'''
    if (HERE/'results/followup_selected.json').exists():
        from summarize_followup import render
        intro,block=render()
        parts=report.split('\n\n',3)
        report='\n\n'.join([parts[0],parts[1],intro,block,parts[3]])
        report=report.replace('선택된 6개 특징:', '예비 Ridge의 6개 특징(최종 후속 모델은 log_dq_var 한 변수):')
        report=report.replace('log10 타깃은 역변환한 사이클 수로 MAPE를 계산한다.', '예비 모델의 log10 타깃은 역변환하여 평가하고, 최종 후속 Linear는 원본 타깃을 학습한다.')
        report=report.replace('선택 기준은 **학습 그룹 CV 평균 MAPE 최소값**이다. 최종 Ridge 파라미터는', '예비 모델의 선택 기준은 **학습 그룹 CV 평균 MAPE 최소값**이다. 예비 Ridge 파라미터는')
        report=report.replace('## 성능 결과', '## 예비 모델 성능(보존)')
        report=report.replace('B2에서는 선택 모델이 중앙값 기준 모델보다도 나빴다.', 'B2에서 예비 Ridge는 중앙값 기준 모델보다 나빴고, 후속 Linear는 개선됐다.')
        report=report.replace('## 오류 분석과 ESS 해석', '## 예비 모델 오류와 ESS 해석')
        report=report.replace('후속 개선은 B1 내부에서 중복 특징 축소와 정책별 안정성을 확인하고, 단수명 학습 표본을 확보하는 방향이다.', 'B1 내부의 중복 특징 축소와 정책별 안정성 실험은 완료했다. 단수명 학습 표본 확보는 별도의 데이터 수집 과제다.')
        report=report.replace('교차 배치 동일 셀 여부가 미확정이므로', '바코드 기준 교차 배치 중복은 0개지만')
        report=report.replace('.venv/bin/python train.py\n', '.venv/bin/python train.py\n.venv/bin/python robustness.py\n')
        report=report.replace('![예측과 실제 수명]', '![예비 모델 예측과 실제 수명]')
        report=report.replace('`data/features.csv`, `models/selected_model.joblib`: 재현 입력과 저장 모델.', '`data/features.csv`: 재현 입력. `models/followup_model.joblib`: 최종 후속 모델. `models/selected_model.joblib`: 보존된 예비 모델.')
    (HERE/'README.md').write_text(report)
    if (HERE/'results/improvement_selected.json').exists():
        from summarize_improvement import render as render_improvement
        render_improvement()
    print(support.to_string(index=False)); print(pd.DataFrame(ranges).to_string(index=False))

if __name__=='__main__': run()
