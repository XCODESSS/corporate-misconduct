Corporate Misconduct Early Warning System
Project Handoff & Technical Report
Status: End of Logistic Regression Baseline (July 2026)

1. Project Overview
   Objective

Develop a production-quality early warning system capable of predicting future corporate accounting misconduct using linguistic signals extracted from SEC 10-K MD&A sections.

The project is inspired by:

Amin & Aßenmacher (2025)
EMNLP Financial NLP Workshop

This project is not intended to be a reproduction.

The goal is a research-grade, modular, production-quality machine learning pipeline.

2. Overall Architecture
   configs/

data/
raw/
interim/
cleaned/
validated/
processed/
datasets/
features/

reports/
evaluation/
figures/
tables/
hypotheses/

src/
ingestion/
preprocessing/
features/
models/
evaluation/
explainability/
utils/

Strict single responsibility.

Every module performs one job only.

3. Coding Standards

Every source file follows the same template.

Large module docstring

Responsibilities

"This module DOES NOT"

Imports

Logger

Single class

Public API

main()

Every function has:

type hints
logging
docstrings

No script contains hardcoded paths.

Everything comes from

configs/settings.py 4. Dataset Pipeline

Completed.

Pipeline

Raw Data

↓

Ingestion

↓

Cleaning

↓

Validation

↓

Temporal Split

↓

Feature Engineering

↓

Modeling 5. Development Dataset

Development

1994–2018

Rows

41,748

Fraud

1,110

≈2.66%

Held-out test

2019–2022 6. Feature Engineering

Current feature set

negative_density

positive_density

uncertainty_density

litigious_density

strong_modal_density

weak_modal_density

constraining_density

Raw counts also exist

N_Words

N_Negative

...

N_Constraining

Current models intentionally use

ONLY

7 density features

Reason

paper consistency
normalized by document length
easier interpretation
SHAP friendliness 7. Current Modeling Pipeline

Implemented

DummyClassifier

↓

Logistic Regression

Future

Optuna

↓

Calibration

↓

Threshold Optimization

↓

XGBoost

↓

Final Evaluation

↓

SHAP 8. Dummy Classifier Results

Strategy

DummyClassifier(
strategy="prior"
)

Results

ROC

0.500

PR

0.0328

Recall

0

Precision

0

Balanced Accuracy

0.50

Brier

0.0317

Interpretation

Exactly what theory predicts.

Evaluation pipeline verified.

No evidence of data leakage.

9. Logistic Regression

Pipeline

StandardScaler

↓

LogisticRegression

Current settings

solver="lbfgs"

class_weight="balanced"

max_iter=1000

C=1

random_state=42 10. Logistic Regression Results

Balanced

ROC

0.5666

PR

0.0557

Precision

0.0374

Recall

0.5792

F1

0.0697

Balanced Accuracy

0.5446

Brier

0.2525 11. Logistic Regression (No Class Weight)

Settings

class_weight=None

Results

ROC

0.5647

PR

0.0561

Precision

0.0794

Recall

0.0042

F1

0.0079

Balanced Accuracy

0.5019

Brier

0.0316

Interpretation

Ranking ability remained almost identical.

Probability calibration became dramatically better.

Recall collapsed because fixed threshold remained unsuitable.

12. Important Finding

The biggest discovery so far

The model itself is not the current bottleneck.

The evaluation policy is.

Current limitations

fixed threshold
no calibration
no hyperparameter optimization

Changing the threshold alone will not fully solve the issue because the balanced model produces poorly calibrated probabilities clustered around 0.5.

13. Prediction Analysis

WalkForwardCV now saves

reports/
evaluation/

        logistic_regression_predictions.csv

Columns

test_year

true_label

predicted_probability

predicted_label

Current balanced model

Mean probabilities

Fraud

0.521

Non Fraud

0.490

Difference

0.031

Very heavy overlap.

Top predictions include multiple false positives.

Possible explanations

Balanced weighting distorts probabilities.

AAER labels are incomplete.

Some false positives may actually represent undiscovered misconduct.

Cannot conclude which explanation is correct.

14. Cross Validation

WalkForwardCV

Implemented.

Current responsibilities

walk forward temporal split
train
evaluate
aggregate metrics
save fold metrics
save prediction probabilities

Uses

predict_proba()

Current metrics

Primary

ROC-AUC

PR-AUC

Threshold metrics

Precision

Recall

F1

MCC

Balanced Accuracy

Calibration

Brier Score 15. Improvements Already Added

WalkForwardCV now

✓ saves prediction probabilities

✓ saves predicted labels

✓ saves true labels

✓ no internal metric rounding

Dummy model

✓ fraud prevalence logging

✓ evaluation years logging

✓ missing feature logging

Logistic Regression

✓ StandardScaler pipeline

✓ modular architecture

✓ identical reporting format

16. Remaining Improvements Inside WalkForwardCV

Still not implemented

Threshold optimization

Current

0.5

Desired

Each fold should learn its own threshold

using

ONLY

training data.

No leakage.

Calibration

Need support for

CalibratedClassifierCV

Future

sigmoid

isotonic
Prediction diagnostics

Need

ROC curves

PR curves

Calibration curves

17. Optuna

Not implemented.

Planned module

src/tuning/

    logistic_optuna.py

Responsibilities

Only

Hyperparameter optimization.

Never

evaluation

feature engineering

threshold tuning

SHAP

Search Space

C

1e-4

↓

100

Log scale.

solver

lbfgs

liblinear

saga
class_weight

None

balanced
max_iter

500

↓

3000

Objective

maximize

PR-AUC

NOT

Accuracy.

NOT

F1.

NOT

ROC.

Reason

Rare-event detection.

Trials

50

↓

100 18. Threshold Optimization

Should NOT be part of Optuna.

Separate module

src/evaluation/

    threshold_optimizer.py

Responsibilities

Search

0.01

↓

0.99

Optimize

One metric

F1

or

MCC

or

Business Cost

Threshold selected

ONLY

using

training folds.

Never

test folds.

Avoid data leakage.

19. Probability Calibration

Planned module

src/evaluation/

    calibration.py

Responsibilities

Platt Scaling

or

Isotonic Regression

Evaluate

Calibration Curve

Brier Score 20. Recommended Future Pipeline
Logistic Regression

↓

Optuna

↓

Best Hyperparameters

↓

Probability Calibration

↓

Threshold Optimization

↓

Held-out Test

↓

Final Evaluation

↓

SHAP 21. XGBoost

Not yet started.

Expected next production model.

Should reuse

WalkForwardCV

Threshold Optimizer

Calibration

Evaluation

Reporting

No duplicated code.

22. Explainability

Planned

SHAP

Only after

best model

is finalized.

23. Research Conclusions So Far

The current linguistic feature set contains predictive signal, but the signal is modest.

Evidence:

Logistic Regression improves ROC-AUC from 0.5000 to 0.5666.
PR-AUC improves from 0.0328 (dummy baseline) to 0.0557, which is a meaningful gain for a dataset with only ~2.7% fraud prevalence.
The current bottleneck is not whether the model can rank filings better than chance; it is how predicted probabilities are converted into decisions and how those probabilities are calibrated.
Evaluation methodology is therefore the highest-priority area for improvement before moving to more complex models. 24. Current Project Status

Overall progress:

Data ingestion: ✅ Complete
Preprocessing: ✅ Complete
Temporal splitting: ✅ Complete
Feature engineering: ✅ Complete
Statistical validation: ✅ Complete
Walk-forward cross-validation: ✅ Complete
Dummy classifier baseline: ✅ Complete
Logistic Regression baseline: ✅ Complete
Prediction persistence: ✅ Complete
Optuna tuning: ⏳ Planned
Probability calibration: ⏳ Planned
Threshold optimization: ⏳ Planned
XGBoost: ⏳ Planned
Final evaluation on held-out test: ⏳ Planned
SHAP explainability: ⏳ Planned

┌─────────────────────────────────────────────────────────────────┐
│ EXPERIMENT ORCHESTRATION │
│ (experiments.yaml, run*pipeline.py, train_models.py) │
└────────────────────────┬────────────────────────────────────────┘
│
┌───────────────┼───────────────┐
↓ ↓ ↓
┌──────────────────┐ ┌─────────────┐ ┌──────────────────┐
│ CONFIGURATION │ │ ARTIFACTS │ │ EXPERIMENT MGT │
│ (Config Layer) │ │ (Caching) │ │ (Tracking) │
├──────────────────┤ ├─────────────┤ ├──────────────────┤
│ • settings.py │ │ • models/ │ │ • optuna/ │
│ • paths.py │ │ • scalers/ │ │ • experiment │
│ • dataset.yaml │ │ • embeddings│ │ branches │
│ • models.yaml │ │ • feature* │ └──────────────────┘
│ • features.yaml │ │ cache/ │
└────────┬─────────┘ └─────────────┘
│ ↑
└────────┬───────────┘
│
┌────────▼─────────────────┐
│ DATA LAYER (src/) │
├──────────────────────────┤
│
┌──────┴──────────────────────┐
│ INGESTION LAYER │
├─────────────────────────────┤
│ • ingest*firm_years.py │ ← Load FinNLP + AAER
│ • ingest_labels.py │ ← Create fraud labels
│ • merge_labels.py │ ← Consolidate
│ • validate_raw_data.py │ ← Schema validation
├─────────────────────────────┤
└────┬──────────────┘
│
┌───────────▼──────────────────┐
│ PREPROCESSING LAYER │
├──────────────────────────────┤
│ • Deduplication │
│ • Text normalization │
│ • Missing value handling │
│ • Data cleaning │
├──────────────────────────────┤
└────┬──────────────┘
│
┌───────────▼──────────────────────────────────────┐
│ FEATURE ENGINEERING LAYER (8 Extractors) │
├────────────────────────────────────────────────────┤
│ │
│ ┌─────────────────┐ ┌──────────────────┐ │
│ │ Text Features │ │ Financial │ │
│ ├─────────────────┤ ├──────────────────┤ │
│ │ • lm_features │ │ • fin_features │ │
│ │ • lexical* │ │ (ratios, trends) │ │
│ │ • readability │ └──────────────────┘ │
│ │ • semantic │ │
│ │ • structural │ ┌──────────────────┐ │
│ │ • behavioral │ │ Embeddings │ │
│ │ • embedding │ ├──────────────────┤ │
│ └─────────────────┘ │ • Word2Vec │ │
│ │ • GloVe │ │
│ │ • FastText │ │
│ │ • Cached vectors │ │
│ └──────────────────┘ │
│ │
│ ┌──────────────────────────────────┐ │
│ │ merge*features.py │ │
│ │ (Feature concatenation & caching)│ │
│ └──────────────────────────────────┘ │
├────────────────────────────────────────────────────┤
└────┬──────────────┘
│
┌───────────▼──────────────────┐
│ DATASET PREPARATION LAYER │
├──────────────────────────────┤
│ • prepare_dataset.py │
│ • Train/val/test splitting │
│ • Feature scaling │
│ • Vectorizer fitting │
│ • Parquet export │
├──────────────────────────────┤
└────┬──────────────┘
│
┌───────────▼──────────────────────────────┐
│ MODEL TRAINING LAYER (7 Models) │
├───────────────────────────────────────────┤
│ │
│ Baseline Models Advanced Models │
│ ├─ dummy_classifier ├─ xgboost │
│ ├─ logistic* ├─ lightgbm │
│ │ regression ├─ catboost │
│ └─ random*forest └─ neural* │
│ network │
│ │
│ Cross-Validation: WalkForwardCV │
│ └─ Expanding-window by year │
│ └─ Temporal consistency │
├───────────────────────────────────────────┤
└────┬──────────────┘
│
┌───────────▼──────────────────────────────┐
│ EVALUATION & ANALYSIS LAYER │
├───────────────────────────────────────────┤
│ │
│ ├─ cross_validation.py │
│ │ └─ WalkForwardCV class │
│ │ └─ Fold generation & metrics │
│ │ │
│ ├─ metrics.py │
│ │ └─ ROC-AUC, F1, Precision, Recall │
│ │ └─ Brier, MCC, Balanced Acc │
│ │ │
│ ├─ feature_importance.py │
│ │ └─ Model-specific importance │
│ │ │
│ ├─ shap_analysis.py │
│ │ └─ SHAP values & force plots │
│ │ │
│ ├─ calibration.py │
│ │ └─ Probability calibration │
│ │ │
│ ├─ statistical_tests.py │
│ │ └─ Hypothesis testing (t-tests, etc)│
│ │ └─ Correlation matrix │
│ │ └─ VIF (multicollinearity) │
│ │ │
│ ├─ error_analysis.py │
│ │ └─ Misclassification patterns │
│ │ │
│ └─ analyze_prediction.py │
│ └─ Per-sample prediction analysis │
│ │
├───────────────────────────────────────────┤
└────┬──────────────┘
│
┌───────────▼──────────────────────────────┐
│ REPORTING & PERSISTENCE LAYER │
├───────────────────────────────────────────┤
│ • fold_results.csv │
│ • cv_summary.json │
│ • predictions.csv │
│ • statistical_report.json │
│ • Hypothesis test results │
│ • SHAP importance scores │
└───────────────────────────────────────────┘
This is the actual structure
