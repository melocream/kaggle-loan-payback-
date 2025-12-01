# Kaggle Playground Series S5E11 - Loan Payback Prediction

## Result

| Metric | Score |
|--------|-------|
| Private LB | **0.92856** |
| Public LB | 0.92741 |
| Rank | **354th / 3,726 teams (Top 9.5%)** |
| Medal | Bronze |

## Competition Overview

Binary classification task to predict whether a loan will be paid back.

- Competition: [Playground Series S5E11](https://www.kaggle.com/competitions/playground-series-s5e11)
- Evaluation Metric: AUC-ROC
- Period: Nov 2025

## Approach

### Feature Engineering

- **Financial Ratios**: income_loan_ratio, debt_burden, affordability, default_risk
- **ROUND Features**: annual_income, loan_amount at different rounding levels (1000s, 100s, 10s)
- **Credit Tiers**: FICO tier, VantageScore tier mappings
- **Interactions**: 2-way and 3-way categorical interactions
- **Target Encoding**: cuml.preprocessing.TargetEncoder (GPU accelerated)
- **Count Encoding**: Frequency-based features
- **Original Data Features**: Mean/Count aggregations from original dataset

### Modeling

| Model | OOF AUC |
|-------|---------|
| XGBoost | 0.92755 |
| CatBoost | 0.92761 |
| **Blend** | **0.92801** |

**Key Techniques:**
- XGBoost + CatBoost Ensemble (0.4 : 0.6 blend)
- Seed Ensemble (4 seeds: 42, 123, 456, 789) - 분산 감소 효과
- Pseudo-labeling (confidence > 0.95) - Round 2 학습
- 8-Fold Cross Validation

### Model Parameters

**XGBoost:**
- tree_method: gpu_hist
- grow_policy: lossguide
- max_leaves: 128
- learning_rate: 0.01

**CatBoost:**
- depth: 6
- iterations: 15000
- learning_rate: 0.02
- task_type: GPU

## What I Learned (배운 점)

1. **Target Encoding inside CV loop** - leakage 방지를 위해 중요
2. **Seed Ensemble** - 단일 시드 대비 분산 감소, 안정적인 예측
3. **Pseudo-labeling** - 확신도 높은 예측을 추가 학습 데이터로 활용
4. **Feature Importance 분석** - employment_status가 가장 중요한 변수
5. **CV-LB gap** - OOF 점수와 Public LB 차이 경험

## Files

- `s5e11-loan-payback-v13-seed-ensemble.ipynb` - Final submission notebook

## Environment

- Python 3.10
- RAPIDS cuML (GPU Target Encoding)
- XGBoost, CatBoost, LightGBM
- Kaggle GPU (Tesla P100)
