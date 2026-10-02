# 재현 입력과 특징 정의

`features.csv`는 과제의 세 MAT 파일에서 추출한 139개 셀의 초기 특징이다. Batch 1/2/3는 각각 46/47/46셀, 수명 유효 셀은 46/39/44셀이다. Batch 2 수명 결측 8셀과 Batch 3 결측 2셀은 평가에서 제외한다. 원본 데이터는 [Kaggle](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 제공하며 varcharge 파일은 사용하지 않는다.

| 특징 | 정의 |
| --- | --- |
| log_dq_var | 실제 cycle100의 Qdlin − cycle10의 Qdlin을 전압 포인트 전체에서 분산(ddof=0) 계산 후 log10 변환 |
| QD_slope_100 | cycle2–100의 유효 QDischarge에 대한 사이클별 선형 회귀 기울기(Ah/cycle) |
| chargetime_mean_100 | cycle2–100의 충전 시간 평균(원본 단위) |
| Tavg_mean_100 | cycle2–100의 평균 온도 평균(°C) |
| IR_mean_100 | cycle2–100의 유효 내부 저항 평균(원본 단위) |
| charge_I_time_std_5 | cycle2–5의 충전 전류를 양수 시간 간격으로 가중한 표준편차(A) |

용량 ≤0 또는 >1.3Ah, IR ≤0은 결측으로 취급한다. 전류 특징은 구간 양 끝 전류가 모두 >0.1A, dt>0인 구간만 사용한다. 구간 전류는 양 끝 평균, 가중치는 dt/전체 dt다. 2–5사이클 모두 유효 충전 구간이 있어야 계산한다. 학습 시 결측 대체는 각 학습 fold의 중앙값으로 수행한다.

`cell_id`, `batch`, `policy`, `cycle_life`, `possible_censored`는 관리·분할·정답·품질 메타데이터이며 예측 특징이 아니다. 측정 종료 경고는 마지막 QD>0.885Ah이고 cycle_life=관측 사이클 수+1인 경우다. 전체 기록을 이용하는 이 경고는 EOL 정답 품질을 판정하는 학습 제외 조건이며, 신규 셀의 추론에는 필요하지 않다. 원본 타깃을 임의로 보정하지 않는다.

원본 MAT에서 재추출하려면 아래처럼 실행한다. 원본 파일은 `raw/`에 놓으며 저장소에 포함하지 않는다.

```sh
.venv/bin/python extract_features.py --raw-dir raw --output tmp/reextracted_features.csv
.venv/bin/python verify.py --raw-dir raw
```

필요한 파일명은 `2017-05-12`, `2018-02-20`, `2018-04-12`로 시작한다. 원본을 직접 추출한 입력과 제공 CSV는 셀·특징별로 비교하여 일치를 확인했다. 전체 열화 속도·knee·종료 용량·관측 길이는 CSV의 예측 특징에 포함하지 않는다.

바코드·채널 재확인은 `.venv/bin/python audit_identity.py --raw-dir raw`로 수행한다. 문자열 객체 해독 결과와 B3 공개 코드의 순차 제외 규칙 대응은 `results/cell_identity.csv`, `B3_quality_cohort.csv`에 있다. 식별자·품질 정보는 예측 특징으로 사용하지 않는다. 최종 모델은 `log_dq_var` 한 변수만 사용하므로 추론 CSV에는 그 특징만 있어도 된다.
