"""Metric calculation, threshold analysis, and experiment-result formatting for Phase 4.

Nothing here touches the TEST split -- every function operates on whatever y_true /
y_proba arrays the caller passes in, and Phase 4 notebook cells only ever pass
validation data.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                              f1_score, precision_recall_curve, precision_score,
                              recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def evaluate_at_threshold(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["threshold"] = threshold
    return metrics


def run_experiment(name: str, model_family: str, feature_set_name: str, weighting: str,
                    pipeline, X_train, y_train, X_val, y_val, threshold: float = 0.5) -> dict:
    """Fit on TRAIN, score on VALIDATION only. Returns (result_record, fitted_pipeline, y_proba)."""
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time_s = time.time() - t0

    t0 = time.time()
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    inference_time_s = time.time() - t0

    metrics = evaluate_at_threshold(y_val, y_proba, threshold=threshold)
    record = {
        "experiment": name,
        "model": model_family,
        "features": feature_set_name,
        "weighting": weighting,
        "train_time_s": round(train_time_s, 2),
        "inference_time_s": round(inference_time_s, 3),
        **metrics,
    }
    return record, pipeline, y_proba


def threshold_table(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Coarse, display-friendly precision/recall/F1 trade-off table."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = [evaluate_at_threshold(y_true, y_proba, threshold=t) for t in thresholds]
    return pd.DataFrame(rows)


def best_f1_threshold(y_true, y_proba) -> dict:
    """Exact best-F1 operating point using the fine-grained thresholds scikit-learn's
    precision_recall_curve already computes from the observed score distribution,
    rather than a coarse manually-chosen grid.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = np.where((precision + recall) > 0,
                   2 * precision * recall / (precision + recall + 1e-12), 0.0)
    f1 = f1[:-1]  # precision_recall_curve returns one more point than thresholds
    best_idx = int(np.argmax(f1))
    return evaluate_at_threshold(y_true, y_proba, threshold=float(thresholds[best_idx]))


def pr_curve_points(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds


def roc_curve_points(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


def segment_performance(df: pd.DataFrame, segment_col: str, fraud_col: str = "fraud",
                         pred_col: str = "pred") -> pd.DataFrame:
    """Precision/recall per segment, with explicit tp/fp/fn/n_flagged counts so a
    segment with zero fraud (precision/recall undefined -> NaN) doesn't get silently
    conflated with a segment that has genuine false positives.
    """
    d = df[[segment_col, fraud_col, pred_col]].copy()
    d["tp"] = (d[pred_col] == 1) & (d[fraud_col] == 1)
    d["fp"] = (d[pred_col] == 1) & (d[fraud_col] == 0)
    d["fn"] = (d[pred_col] == 0) & (d[fraud_col] == 1)
    g = d.groupby(segment_col, observed=True).agg(
        n=(fraud_col, "size"), n_fraud=(fraud_col, "sum"), n_flagged=(pred_col, "sum"),
        tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
    g["precision"] = np.where(g["n_flagged"] > 0, g["tp"] / g["n_flagged"], np.nan)
    g["recall"] = np.where(g["n_fraud"] > 0, g["tp"] / g["n_fraud"], np.nan)
    return g
