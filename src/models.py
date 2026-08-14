"""Model factory functions and pipeline construction for Phase 4.

Every factory takes an explicit random_state (defaulting to the module-level
RANDOM_STATE) so re-running the notebook is reproducible. Preprocessing lives in
src/features.py :: build_preprocessor(); this module only builds the estimator and
wires it to a preprocessor via build_pipeline().
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 42


def build_pipeline(preprocessor, estimator) -> Pipeline:
    """Wire a fitted-per-call preprocessor (Phase 3) to an estimator (Phase 4)."""
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def make_dummy(strategy: str = "stratified", random_state: int = RANDOM_STATE) -> DummyClassifier:
    """No-skill reference. 'stratified' predicts according to the training class
    prior rather than always predicting the majority class, so precision/recall/PR-AUC
    are all non-degenerate and PR-AUC lands at the fraud prevalence, as expected for any
    ranking with no real signal.
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)


def make_logistic_regression(class_weight: str | None = None,
                              random_state: int = RANDOM_STATE) -> LogisticRegression:
    return LogisticRegression(max_iter=3000, solver="lbfgs", class_weight=class_weight,
                               random_state=random_state)


def make_random_forest(class_weight: str | None = None, random_state: int = RANDOM_STATE,
                        n_estimators: int = 200, max_depth: int | None = None,
                        min_samples_leaf: int = 5, n_jobs: int = -1) -> RandomForestClassifier:
    """~595k rows: 200 trees with a min-leaf-size floor keeps training under a minute
    and avoids single-transaction leaves that would just memorize the rare positive class.
    """
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf, class_weight=class_weight,
                                   random_state=random_state, n_jobs=n_jobs)


def compute_scale_pos_weight(y) -> float:
    """XGBoost's imbalance-aware analogue of class_weight='balanced': ratio of negatives
    to positives in the TRAINING labels only.
    """
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))


def make_xgboost(scale_pos_weight: float = 1.0, random_state: int = RANDOM_STATE,
                  n_estimators: int = 200, max_depth: int = 6, learning_rate: float = 0.1,
                  n_jobs: int = -1) -> XGBClassifier:
    return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                          learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", scale_pos_weight=scale_pos_weight,
                          random_state=random_state, n_jobs=n_jobs, tree_method="hist")
