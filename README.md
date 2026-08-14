# Fraud Detection Case Study

A complete, evidence-driven fraud detection case study — from data audit through a
locked, tested model and business recommendations. Built as a technical case study for a
Deloitte interview process. See
[Requirements/Interview_Case_Study.docx](Requirements/Interview_Case_Study.docx) for the
original brief.

**Status: complete.** All six phases (data understanding → EDA → validation design &
feature engineering → model experiments → final tuning & locked test evaluation →
delivery) are finished. The model has been evaluated on the held-out test set exactly
once, and that result is final.

## Business objective

Detect fraudulent card transactions in a highly imbalanced dataset (~1.2% fraud) using
only information that would realistically be available at the moment a transaction is
scored, and turn the result into a defensible, business-usable recommendation — not just
a model.

## Dataset overview

- 594,643 transactions, 7,200 confirmed fraud (1.21% prevalence, ~82:1 imbalance).
- 180 hourly time steps (~7.5 days).
- Fields: `step`, `customer`, `age`, `gender`, `merchant`, `category`, `amount`,
  `zipcodeOri`, `zipMerchant` (both constant, dropped), `fraud` (target).
- No missing values, no duplicate rows. Customers and merchants recur throughout the
  whole window; fraud rate and category mix both drift over it (1.52% → ~1.08% fraud
  rate from day 0 to the test period).
- Dataset placed at `Data/Input/fraud.csv` (not committed — see Setup below).

## Analytical methodology

Chronological, not random, train/validation/test split — because entities recur across
the whole window and fraud/category composition drift over time, a random split would
both leak future customer behavior into training and evaluate on a blended mix of time
periods rather than a genuine future one.

| Split | Step range | Transactions | Fraud | Fraud rate |
|---|---:|---:|---:|---:|
| Train | 0–119 | 374,914 | 4,800 | 1.280% |
| Validation | 120–143 | 86,498 | 960 | 1.110% |
| Test | 144–179 | 133,231 | 1,440 | 1.081% |

This is a realistic *future-period* evaluation for this dataset — not a claim that it
eliminates every possible form of leakage.

## Leakage prevention

Every customer-behavioral feature is built from transactions with `step` **strictly
less than** the current transaction's step — never `<=`. This distinction is not
cosmetic: 7.31% of rows share a `(customer, step)` pair (multiple transactions from the
same customer within the same hour), and the dataset provides no reliable ordering within
an hour. Transactions sharing a step therefore cannot leak into each other's historical
features in either direction; every row in the same step gets identical, same-step-blind
history. Verified by (a) a brute-force recomputation of historical features on a random
sample of rows, and (b) an explicit check that same-step transactions share identical
feature values. No feature — behavioral or otherwise — ever uses the `fraud` label from
any transaction (no target encoding).

## Feature engineering

- **Feature Set A** (transaction-level, 6 features): `amount`, `category`, `merchant`,
  `age`, `gender`, `is_enterprise`.
- **Feature Set B** (selected, 16 features): Set A + 10 leakage-safe customer-behavioral
  features (prior transaction count, mean/median/std/max/total amount, amount relative to
  the customer's own prior mean/median, time since last transaction, first-transaction
  flag).
- **Feature Set C** (evaluated, not selected, 19 features): Set B + 3 non-target
  merchant-history features — rejected, see Model Selection below.
- Cold-start (a customer's first transaction) is handled with an explicit indicator flag
  plus median imputation fit on the training fold only, inside the same scikit-learn
  pipeline used for scoring — no separate, easy-to-desync imputation logic.

## Models evaluated

Logistic Regression, Random Forest, and XGBoost, each unweighted and class-weighted,
across all three feature sets (an 18-run controlled grid), plus a no-skill dummy baseline
and two sanity baselines (amount-only, amount+category).

## Imbalance strategy

The ~1.2% prevalence was treated as an empirical modeling question, not a reason to
reach for resampling by default. Class weighting was tested and did **not** improve
threshold-independent ranking quality (Average Precision) or the best achievable
validation F1 after its own threshold was independently optimized — it traded precision
for recall at a fixed threshold without producing a better underlying model. SMOTE was
considered and explicitly **not** applied: the unweighted model already showed strong
minority-class discrimination, weighting hadn't demonstrated an unresolved ranking
problem for oversampling to fix, and standard SMOTE has a real technical complication
here (interpolating one-hot categorical vectors produces meaningless fractional
encodings). The final model is trained on the natural class distribution.

## Model selection

XGBoost + Feature Set B, unweighted. Selected on validation evidence:

- Highest validation PR-AUC (0.9464) among Logistic Regression, Random Forest, and
  XGBoost, across all three feature sets and both weighting strategies.
- ~10x faster to train than the closest Random Forest configuration (0.9336 PR-AUC).
- Feature Set C (merchant history) rejected: for XGBoost, PR-AUC was flat-to-slightly-negative
  vs. Set B — added state and complexity without measurable benefit, not evidence of
  overfitting.
- A bounded, 12-configuration hyperparameter search around the baseline found nothing
  clearing a "meaningful improvement" bar (best alternative: +0.0011 PR-AUC, +0.0007 F1,
  at ~2x the trees) — the baseline configuration was retained deliberately, not left
  unfinished.

## Final test results (unseen future period, evaluated once)

| Metric | Validation | **Test** |
|---|---:|---:|
| PR-AUC | 0.9464 | **0.9482** |
| ROC-AUC | 0.9990 | **0.9991** |
| Precision | 91.32% | **91.29%** |
| Recall | 85.52% | **83.75%** |
| F1 | 0.8833 | **0.8736** |
| Accuracy | 99.75% | **99.74%** |

Confusion matrix at the locked threshold (0.5631, F1-maximizing on validation only — a
technical benchmark, not a claimed economically optimal production threshold):

|  | Predicted legit | Predicted fraud |
|---|---:|---:|
| **Actual legit** | 131,676 | 115 |
| **Actual fraud** | 234 | 1,206 |

Of 1,440 fraud cases in the test period: **1,206 detected, 234 missed**. Of 1,321 total
alerts generated: **91.3% corresponded to real fraud**.

## Business interpretation

- **Deploy as risk-prioritization, not automatic blocking.** Route high-risk transactions
  to human investigation; ~8.7% of alerts are false positives, and even confident false
  positives resemble genuine fraud closely.
- **Category, amount, and merchant are the model's dominant signals** (confirmed via
  permutation importance and native TreeSHAP — the built-in XGBoost gain importance was
  found to overstate merchant's share due to its higher one-hot cardinality and was
  superseded). Customer behavioral history adds real, smaller, incremental value.
- **Low-amount fraud (<~$100) is the main blind spot** — recall falls to ~43-45% under
  $50, vs. ~97-100% above $200, confirmed on two independent time periods. Complementary
  controls are recommended for this segment.
- **The production threshold should be set from business inputs** (investigation
  capacity, cost of missed fraud vs. false positives) not available in this case study —
  0.5631 is a technical validation benchmark only.

## Limitations

Short observation window (~7.5 days, no real seasonality); some data characteristics
suggest a possibly simulated/structured source (stated cautiously, not asserted); a few
categories have zero observed fraud and are untested; no business cost data was available
for threshold optimization; the behavioral features require reliable low-latency
customer-history infrastructure in production; almost no evidence exists on genuinely new
customers/merchants since nearly all recur across splits; and no fairness analysis was
performed — none should be inferred.

## Repository structure

```
fraud-detection-case-study/
│
├── README.md
├── requirements.txt                        # exact pinned versions (see Reproducibility)
├── .gitignore
│
├── notebooks/
│   └── 01_fraud_detection_analysis.ipynb   # full analytical narrative, all 6 phases
│
├── src/
│   ├── __init__.py
│   ├── data.py                             # loading, cleaning, chronological split
│   ├── features.py                         # leakage-safe feature engineering, preprocessing
│   ├── validation.py                       # leakage / split-integrity assertions
│   ├── models.py                           # model factories (LogReg, RF, XGBoost, Dummy)
│   └── evaluation.py                       # metrics, threshold analysis, experiment runner
│
├── models/
│   ├── final_model_xgboost_setB.joblib     # locked, fitted pipeline (preprocessing + model)
│   ├── final_model_config.json             # features, hyperparameters, threshold, metrics
│   └── final_metrics_summary.csv           # corrected dummy baselines + final test metrics
│
├── Data/
│   ├── Input/                              # place fraud.csv here (gitignored, not redistributed)
│   └── Output/figures/                     # all exported chart PNGs (gitignored)
│
├── Requirements/
│   └── Interview_Case_Study.docx           # original case brief
│
└── presentation/
    ├── Fraud_Detection_Case_Study.pptx
    ├── Fraud_Detection_Case_Study.pdf
    
```

**Note on structure:** figures live under `Data/Output/figures/` rather than a top-level
`outputs/` folder. This keeps the path the notebook has used and validated across all six
phases; renaming it would require re-executing the full model-training notebook
end-to-end purely for a cosmetic path change, which was judged not worth the risk this
late in the project. Flagged here rather than done silently.

## Setup

```
pip install -r requirements.txt
```

Place the raw dataset at `Data/Input/fraud.csv`. It is not committed to version control —
redistribution rights for the source data were not confirmed, so it is excluded from
GitHub; obtain it from the original case study source and place it at that path.

## Running the analysis

Open `notebooks/01_fraud_detection_analysis.ipynb` and run all cells top to bottom, or:

```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_fraud_detection_analysis.ipynb
```

The notebook is self-contained and deterministic (`RANDOM_STATE=42` throughout) —
re-running it reproduces every number and figure referenced here. Runtime is a few
minutes, dominated by the Phase 4/5 model grids.

## Reproducibility

- All splits are chronological (`src/data.py`) — see Analytical Methodology above.
- All feature engineering is leakage-safe by construction and checked by assertions in
  `src/validation.py`.
- The TEST split is loaded into the notebook exactly once (Phase 5), after the model,
  features, and threshold were already locked using train/validation only.
- `requirements.txt` pins the **exact versions of the environment that actually produced
  every result in this repository** (Anaconda Python 3.10.9: `pandas==1.5.3`,
  `numpy==1.26.4`, `scikit-learn==1.2.1`, `xgboost==1.7.6`, `matplotlib==3.7.0`,
  `seaborn==0.12.2`). This matters more than usual here: scikit-learn's own pickle format
  is not guaranteed compatible across even minor versions, and `models/final_model_xgboost_setB.joblib`
  was confirmed to reload and reproduce identical test predictions only when the pinned
  versions are used — a mismatched `scikit-learn` version (e.g. 1.2.2) raised an
  `AttributeError` on unpickling in testing. Use the pinned versions.
- To independently verify the saved model without re-running the notebook:

  ```python
  import joblib, json
  cfg = json.load(open("models/final_model_config.json"))
  model = joblib.load("models/final_model_xgboost_setB.joblib")
  # model.predict_proba(X) where X has the columns in cfg["features"]
  # classify fraud where predict_proba[:, 1] >= cfg["locked_threshold"]
  ```
