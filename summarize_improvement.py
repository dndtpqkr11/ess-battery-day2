"""Generate the current submission README, improvement report and figures."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mini-project-day2-mpl')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def render():
    meta=json.loads((HERE/'results/improvement_selected.json').read_text())
    ev=pd.read_csv(HERE/'results/improvement_evaluations.csv')
    cur=ev[ev.model=='Improved'].set_index('split')
    prev=ev[ev.model=='Previous'].set_index('split')
    summary=pd.read_csv(HERE/'results/improvement_summary.csv')
    scores=pd.read_csv(HERE/'results/improvement_scores.csv')
    pred=pd.read_csv(HERE/'results/improvement_predictions.csv')
    cv=meta['fixed_cv_MAPE_pct'];vm=cur.loc['Valid_B1_reused','MAPE_pct']
    tm=cur.loc['Test_B2_reused','MAPE_pct'];b3=cur.loc['Additional_B3_reused','MAPE_pct']
    intro=f'''최종 모델은 **초기 ΔQ 분산 한 변수의 MAPE 가중 선형 회귀**다. 학습 28셀에서 정책별 20회 분할로 10개 후보를 비교했다. 선택 평균 MAPE는 {meta['mean_repeated_holdout_MAPE_pct']:.2f}%로, 이전 Linear 8.69%보다 낮다. Batch 2 재사용 평가 MAPE는 **{tm:.2f}%**(이전 28.72%)다. 목표 9.1%는 미달이며 Hold-out과 Batch 3은 소폭 악화됐다. 이미 본 자료의 재평가이므로 새 독립 테스트 성능으로 주장하지 않는다.'''
    table=f'''| 구분 | MAPE (%) / Gap (%p) | 비고 |
| --- | ---: | --- |
| Train (Batch 1 CV) | {cv:.2f} ± {meta['fixed_cv_std_pct']:.2f} | 고정 3-fold; 선택 후 진단 |
| Valid (Batch 1 Hold-out) | {vm:.2f} | 8셀, 선택에 미사용; 이전에 본 자료 |
| Test (Batch 2) | {tm:.2f} | 39셀, 재사용 평가 |
| Gap (Train-Valid) | {vm-cv:+.2f} | Valid−Train, %p |
| Gap (Valid-Test) | {tm-vm:+.2f} | Test−Valid, %p |
| Gap (Target-Test) | {tm-9.1:+.2f} | Test−9.1, %p |

Batch 3 추가 재사용 평가: **{b3:.2f}%**(44셀). MAPE는 셀별 상대 오차의 평균이고 Gap은 퍼센트포인트다. CV 표준편차는 3개 fold의 산포이며 신뢰구간이 아니다.
'''
    compare='''| 모델 | B1 Hold-out | B2 재사용 | B3 재사용 |
| --- | ---: | ---: | ---: |
| 예비 Ridge, 6변수 | 9.84% | 61.25% | 15.67% |
'''
    compare+=f"| 이전 Linear, 1변수 | {prev.loc['Valid_B1_reused','MAPE_pct']:.2f}% | {prev.loc['Test_B2_reused','MAPE_pct']:.2f}% | {prev.loc['Additional_B3_reused','MAPE_pct']:.2f}% |\n"
    compare+=f'| MAPE 가중 Linear, 1변수 | {vm:.2f}% | {tm:.2f}% | {b3:.2f}% |\n'
    nested=f'''모델 후보를 고른 동일 분할의 점수는 선택 편향을 포함한다. 이를 점검하려고 **3개 바깥 정책 fold 각각의 학습 부분에서 20회 분할·10개 후보 선택을 다시 수행**했다. 바깥 검증 셀·정책은 내부 선택과 전처리에 들어가지 않는다. 이 선택 절차의 바깥 CV 평균 MAPE는 **{meta['nested_cv_MAPE_pct']:.2f} ± {meta['nested_cv_std_pct']:.2f}%**, 같은 바깥 fold의 이전 Linear는 {meta['nested_previous_MAPE_pct']:.2f}%다. 바깥 fold의 선택 모델은 서로 다르다. 이는 최종 고정 모델의 CV {cv:.2f}% 및 후보 선택용 20회 평균과 다른 지표다. B1 데이터와 기존 분할 자체를 이전 작업에서 확인한 이력이 있으므로 이 역시 내부 진단이며 새 독립 검증이 아니다.'''
    b2=pred[(pred.model=='Improved')&(pred.split=='Test_B2_reused')]
    examples='\n'.join(f'- {r.cell_id}: 실제 {r.actual:.0f}, 예측 {r.predicted:.1f}사이클, APE {r.APE_pct:.1f}%.' for _,r in b2.nlargest(3,'APE_pct').iterrows())
    rows='\n'.join(f'| {r.candidate} | {r.mean_MAPE_pct:.2f} | {r.std_MAPE_pct:.2f} | {r.max_MAPE_pct:.2f} |' for _,r in summary.iterrows())
    detail=f'''# DAY 2 추가 개선 — MAPE 목적함수와 선택 절차 검증

{intro}

## 변경 이유와 구현

이전 Linear는 제곱 오차를 최소화했지만 과제는 MAPE를 평가한다. 정답 y가 양수이면 MAPE 목적함수는 `sum(abs(y-p)/y)`다. `QuantileRegressor(quantile=0.5, alpha=0)`에 각 학습 fold의 `1/y` 가중치를 주면 같은 최소값을 갖는 가중 절대 오차 선형 회귀가 된다. 가중치는 평균 1로 정규화한다. 검증·외부 정답은 가중치나 전처리 계산에 쓰지 않는다. 입력은 기존 `log_dq_var` 하나를 유지했다.

비교 후보는 기존 Linear, 상대 제곱 오차 가중 Linear(1/y²), MAPE 가중 Linear, 비가중 절대 오차 Linear, 원본·log 타깃 Huber, log 타깃 중앙값 회귀, log 타깃 Linear, 원본 타깃 Ridge(alpha=1/10) 총 10개다. 후보 목록·가중치·분할은 외부 재평가 전에 저장했다. 반복 분할 평균 MAPE 최솟값으로 선택했으며 B2 성능으로 후보를 다시 바꾸지 않았다.

| 후보 | 20회 평균 MAPE (%) | 표준편차 | 최대 |
| --- | ---: | ---: | ---: |
{rows}

반복 분할은 셀을 공유한다. 20개의 독립 표본이나 유의성 검정 근거로 해석하지 않는다.

![학습 데이터 후보 비교](figures/improvement_selection.png)

## 모델 선택을 포함한 검증

{nested}

## 성능과 변화

{table}

{compare}

B2 개선은 약 {prev.loc['Test_B2_reused','MAPE_pct']-tm:.2f}%p지만 모든 평가 집합에서 좋아진 결과는 아니다. B1 Hold-out은 {vm-prev.loc['Valid_B1_reused','MAPE_pct']:+.2f}%p, B3는 {b3-prev.loc['Additional_B3_reused','MAPE_pct']:+.2f}%p 변했다. 이번 모델 채택 근거는 B1 내부의 사전 지정 선택 규칙이며, 외부 점수로 선택한 것이 아니다. 이전 모델은 별도 파일로 유지했다.

![평가 집합별 예측](figures/improvement_prediction_scatter.png)

큰 B2 오류 사례:

{examples}

## 추론 진단과 한계

`predict.py --diagnostics`는 특징의 학습 최솟값·최댓값 이탈, 특징 결측 수, 예측 수명의 학습 타깃 범위 이탈을 표시한다. 수명 학습 범위는 534–1054사이클이다. 범위 표시 자체는 예측 구간, 오류 확률, 안전 판정이 아니다. 예측값을 학습 범위로 잘라 성능을 바꾸지 않는다. 실제 수명이 필요 없는 초기 특징 CSV로 실행된다.

단수명 과대 예측과 배치 이동은 남아 있다. 현재 점수로 실제 ESS 교체 시점·안전성·비용 절감을 보장할 수 없다. 현장 적용에는 단수명 표본과 별도 배치 검증, SOH 감시가 필요하다.

## 재현·검증

`improve.py`는 200개 기본 후보·분할 학습과 600개 내부 중첩 후보·분할 학습을 수행한다. `verify_improvement.py`가 800개를 다시 학습하고 정책 분리·선택·저장 예측·평가 지표·Gap을 대조한다. 선택된 MAPE 가중 모델의 학습 목적함수 최솟값은 별도 원본 입력 선형계획법으로 확인한다. 기존 예비·후속 모델과 예측 해시, DAY 1 파일 40개 해시도 대조한다. 기계 기록은 `results/improvement_validation.json`이다.

```sh
python improve.py
python summarize_improvement.py
python verify.py
python predict.py --input data/features.csv --output tmp/predictions.csv --diagnostics
python package.py
```
'''
    (HERE/'IMPROVEMENT.md').write_text(detail)
    readme=f'''# ESS 배터리 수명 예측 — DAY 2

울산 4반 박세웅 (U115) | 초기 100사이클로 총 수명 `cycle_life` 회귀 예측

{intro}

## 프로젝트 개요와 EDA → 구현

데이터는 MIT–Stanford 배터리 자료다. Batch 1(2017-05-12)로 학습·검증하고 Batch 2(2018-02-20)로 필수 평가, Batch 3(2018-04-12)로 추가 평가한다. 셀 139개 중 유효 수명은 129개다.

| DAY 1 발견 | 모델 설계·구현 |
| --- | --- |
| B1의 <550사이클 셀은 1/46 | 분류 대신 총 수명 회귀; 단수명 일반화 한계 명시 |
| 수명 중앙값 B1/B2/B3 = 858.5/472/1005.5 | B1 학습, B2 외부 평가, B3 추가 평가 |
| 말기 열화 가속·knee 후보 관측 | 전체 곡선·knee를 입력에서 제외 |
| log ΔQ 분산–수명 Spearman = −0.871/−0.709/−0.797 | Q100(V)−Q10(V)의 log 분산 사용 |
| 충전 조건–수명 관계는 배치별로 다름 | 정책 문자열은 입력 대신 분할 그룹으로 사용 |
| ΔQ 특징 간 중복 0.971–0.989 | 한 변수 모델과 다변수 정규화 모델 비교 |
| B1 종료 경고 10셀, B2 IR=0인 유효 수명 6셀 | 경고 셀 학습 제외, IR=0은 결측; 최종 모델은 IR 미사용 |

예비 6개 초기 특징의 정의·원본 재추출은 [data/README.md](data/README.md), 이전 실험은 [FOLLOWUP.md](FOLLOWUP.md)에 있다. 이전 문서는 해당 실험 시점의 기록이며 현재 기본 모델은 아래 개선 모델이다.

## 분할·파이프라인·모델 선택

B1 46셀에서 종료 경고 10셀을 제외한 36셀을 정책 그룹 hold-out(seed=42)으로 나눴다. 학습 28셀/15정책, 검증 8셀/5정책이다. 학습·검증 및 CV 양쪽에 셀·정책 교집합이 없다. CV에서도 정책 그룹을 분리한다. 무작위 셀 분리만으로 같은 정책의 양쪽 포함을 막을 수 없기 때문이다.

학습 중앙값 결측 대체 → 표준화 → 회귀 모델을 각 fold 안에서 fit한다. 최종 입력은 `log_dq_var` 하나다. 전체 관측 길이, 미래 열화, knee, 배치·셀 ID, 정책은 입력이 아니다. B2/B3는 fit·선택에 쓰지 않았다. 타깃 결측 10셀은 평가에서 제외하고 예측만 저장한다.

예비 단계에서 DummyMedian, Linear, Ridge, ElasticNet, 얕은 RandomForest와 원본/log 타깃을 비교했다. 특징 축소 후 추가 개선에서는 고정된 10개 후보를 정책별 20회 분할로 비교했다. 최종 `QuantileRegressor(quantile=0.5, alpha=0)`는 `1/y` 가중 절대 오차로 MAPE와 같은 목적함수를 최소화한다. 가중치도 해당 fold의 학습 정답만으로 계산한다. 상세 선택 근거와 전체 후보는 [IMPROVEMENT.md](IMPROVEMENT.md)에 있다.

{nested}

## 성능 결과

{table}

{compare}

![최종 모델 예측](figures/improvement_prediction_scatter.png)

## 오류 분석과 ESS 해석

B2 MAPE는 이전 모델보다 {prev.loc['Test_B2_reused','MAPE_pct']-tm:.2f}%p 낮지만 Hold-out·B3은 소폭 악화됐다. 목표 대비 차이는 {tm-9.1:.2f}%p다. 기존 수명 학습 범위는 534–1054사이클이고 B2 39셀 중 30셀은 이보다 짧다. 짧은 수명을 과대 예측하는 문제가 남는다.

{examples}

작은 표본·학습 수명 범위 밖의 셀·정책 및 특징 분포 이동은 오류 원인 가설이다. 인과 설명으로 확정하지 않는다. 예비 6변수 Ridge의 IR 결측·충전 시간 범위 이탈 진단은 [FOLLOWUP.md](FOLLOWUP.md)에 보존했다.

ESS 셀 선별·교체 계획의 후보 정보로 사용할 수 있지만, 단수명 과대 예측은 교체를 늦출 수 있다. 운영 적용에는 해당 화학계·온도·부하의 별도 검증과 단수명 표본, 예측 불확실성 평가·SOH 감시가 필요하다. CLI의 범위 이탈 표시는 이러한 검토를 돕는 진단이며 신뢰구간이나 안전 판정이 아니다.

## 환경 설정·재현

Python 3.11 및 `requirements.txt`의 고정 버전을 사용한다. 이 폴더를 저장소 루트로 올린다. `data/features.csv`로 원본 MAT 없이 재현 가능하다.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/python robustness.py
.venv/bin/python improve.py
.venv/bin/python summarize.py
.venv/bin/python verify.py
.venv/bin/python predict.py --input data/features.csv --output tmp/predictions.csv --diagnostics
.venv/bin/python package.py
```

상위 프로젝트에서는 `.venv/bin/python day2/improve.py`처럼 실행한다. 기본 추론 모델은 `models/improved_model.joblib`다. 이전 모델은 `--model models/followup_model.joblib` 또는 `--model models/selected_model.joblib`로 지정할 수 있다. `--diagnostics`는 학습 범위 메타데이터가 있는 새 모델에서 지원한다.

## 파일·검증·제출

- `extract_features.py`, `data/features.csv`: 원본 특징 추출·재현 입력.
- `train.py`, `robustness.py`, `improve.py`: 예비·특징 축소·MAPE 개선 실험.
- `models/`: 세 단계의 저장 모델. 기존 모델·예측은 해시로 보존 확인.
- `results/`, `figures/`: 분할·후보·선택·예측·성능·검증 결과와 그림.
- `predict.py`: 초기 특징만으로 추론, 선택적 학습 범위 진단.
- `verify.py`, `verify_improvement.py`: 독립 재학습·전처리·정책 분리·목적함수·지표 검증.
- [VALIDATION.md](VALIDATION.md), [IMPROVEMENT.md](IMPROVEMENT.md): 과제 대응과 추가 개선 상세.

DAY 1 제출 파일 40개는 변경하지 않는다. DAY 2 제출물은 공개 GitHub 링크이며 반별 Slack thread로 제출한다. 로컬 ZIP은 업로드 준비용이고 공개·제출은 별도다.

## 출처·팀 구성

[과제](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb), [데이터](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle), [논문 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation). Severson et al. (2019), *Nature Energy* 4, 383–391, DOI: 10.1038/s41560-019-0356-8. 파일 버전·분할 차이 때문에 논문의 정확한 재현으로 해석하지 않는다. 바코드 139개는 모두 고유하다.

박세웅: EDA, 특징 설계, 모델 개발, 평가 및 결과 검토.
'''
    (HERE/'README.md').write_text(readme)
    historical=HERE/'FOLLOWUP.md'
    note='> 이전 개선 단계의 기록입니다. 현재 기본 모델과 실행 방법은 [README.md](README.md), 최신 실험은 [IMPROVEMENT.md](IMPROVEMENT.md)를 참고하세요.\n\n'
    if not historical.read_text().startswith(note):
        historical.write_text(note+historical.read_text())
    marker='\n## 추가 개선 검증 (MAPE 가중 회귀)'
    p=HERE/'VALIDATION.md'
    base=p.read_text().split(marker)[0]
    p.write_text(base+marker+f'''

현재 기본 모델은 `improved_model.joblib`다. 이전 두 모델과 예측은 보존했다. B1 학습 28셀에서 20회 정책 분할·10개 후보로 선택하고 바깥 3-fold 각각에서 선택을 반복했다. 기본 200개·중첩 내부 600개 조합을 독립 재학습한다. 최종 모델의 MAPE 목적함수 최솟값은 별도 선형계획법으로 검증한다.

고정 CV {cv:.2f}±{meta['fixed_cv_std_pct']:.2f}%, Hold-out {vm:.2f}%, B2 재사용 {tm:.2f}%, B3 재사용 {b3:.2f}%. 선택 절차의 바깥 CV {meta['nested_cv_MAPE_pct']:.2f}%. B2 개선과 Hold-out/B3 악화를 함께 보고한다. 목표 9.1%는 미달이며 외부 자료는 재사용 평가다.

`predict.py --diagnostics`는 초기 특징의 학습 범위 이탈·결측과 예측 타깃 범위 이탈을 기록한다. 기존 모델 지정도 지원한다. 재현 순서: `train.py` → `robustness.py` → `improve.py` → `summarize.py` → `verify.py` → `package.py`. 실행 검증 기록은 `results/improvement_validation.json`과 `results/final_validation.json`에 있다.
''')
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,(split,g) in zip(axes,pred[pred.model=='Improved'].groupby('split',sort=False)):
        ax.scatter(g.actual,g.predicted,color='#2563eb',alpha=.75)
        lo=min(g.actual.min(),g.predicted.min())*.9;hi=max(g.actual.max(),g.predicted.max())*1.05
        ax.plot([lo,hi],[lo,hi],'--',color='gray')
        ax.set(title=f'{split}\nMAPE {cur.loc[split,"MAPE_pct"]:.2f}%',xlabel='Actual cycle life',ylabel='Predicted cycle life',xlim=(lo,hi),ylim=(lo,hi))
    fig.tight_layout();fig.savefig(HERE/'figures/improvement_prediction_scatter.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    names=summary.candidate.tolist()
    ax.boxplot([scores[scores.candidate==n].MAPE_pct for n in names],orientation='horizontal',tick_labels=names)
    ax.set(title='10 fixed candidates, 20 B1 training policy splits',xlabel='MAPE (%)')
    fig.tight_layout();fig.savefig(HERE/'figures/improvement_selection.png',dpi=170);plt.close(fig)
    print('Current README, IMPROVEMENT, validation notes and figures updated.')


if __name__=='__main__':render()
