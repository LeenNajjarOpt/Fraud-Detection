"""Leakage-safe feature engineering for the fraud dataset.

Prediction-moment principle (governs everything in this module, see notebooks Phase 3
Section 2): a transaction arriving at time `step` may only use (a) its own fields and
(b) information from transactions strictly BEFORE `step`. No future transactions and no
`fraud` label (current or future) may be used to build any feature here.
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

AMOUNT_COL = "amount"
STEP_COL = "step"


# ---------------------------------------------------------------------------
# A. Deterministic features (current-transaction fields only, no history)
# ---------------------------------------------------------------------------

def add_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived only from a transaction's own fields -- no history, no leakage.

    - is_enterprise: RETAIN. Deterministic flag for the age='U'/gender='E' structural
      pair (Phase 2 Section 9); resolves that collinearity cleanly and Enterprise
      accounts showed a distinct fraud rate (0.59% vs 1.21% overall).
    - day: utility/grouping column for splitting and per-segment reporting only. NOT
      included in any model feature set (see decision in Phase 3 Section 5 -- same
      dataset-specific-drift risk as raw `step`).
    - hour_of_day: EXPERIMENTAL. Phase 2 Section 7 found no genuine hour-of-day cycle
      (the apparent pattern was a day-level artifact). Computed here for optional later
      ablation but excluded from Feature Sets A/B/C.
    """
    df = df.copy()
    df["is_enterprise"] = ((df["age"] == "U") & (df["gender"] == "E")).astype(int)
    df["day"] = df[STEP_COL] // 24
    df["hour_of_day"] = df[STEP_COL] % 24
    return df


# ---------------------------------------------------------------------------
# B. Historical (behavioral) features -- expanding, strictly-prior-step statistics
# ---------------------------------------------------------------------------

def _prior_expanding_stats(df: pd.DataFrame, key_col: str, prefix: str,
                            value_col: str = AMOUNT_COL, step_col: str = STEP_COL) -> pd.DataFrame:
    """
    For every unique (key_col, step_col) pair, compute expanding statistics of value_col
    using ONLY rows with a strictly smaller step for the same key.

    Rows sharing the same (key, step) therefore receive IDENTICAL statistics, and none
    of them contribute to each other's statistics -- this is what prevents same-step
    leakage when no within-step transaction ordering is known (Phase 3 Section 6).

    Returns one row per unique (key_col, step_col) pair with columns:
    {prefix}_tx_count, {prefix}_mean_amount, {prefix}_median_amount, {prefix}_std_amount,
    {prefix}_max_amount, {prefix}_total_amount, {prefix}_last_step.
    Join back onto the full transaction table on [key_col, step_col].
    """
    d = df[[key_col, step_col, value_col]].copy()
    d["_sq"] = d[value_col] ** 2
    d = d.sort_values([key_col, step_col], kind="mergesort")

    g = d.groupby([key_col, step_col], sort=False)
    step_level = g.agg(count=(value_col, "count"), sum=(value_col, "sum"),
                        max=(value_col, "max"), sumsq=("_sq", "sum")).reset_index()
    amounts = g[value_col].apply(list).reset_index(drop=True)
    step_level = step_level.sort_values([key_col, step_col], kind="mergesort").reset_index(drop=True)
    step_level["amounts"] = amounts.values

    gb = step_level.groupby(key_col, sort=False)
    step_level["prior_count"] = gb["count"].cumsum() - step_level["count"]
    step_level["prior_sum"] = gb["sum"].cumsum() - step_level["sum"]
    step_level["prior_sumsq"] = gb["sumsq"].cumsum() - step_level["sumsq"]
    cummax_incl = gb["max"].cummax()
    step_level["prior_max"] = cummax_incl.groupby(step_level[key_col], sort=False).shift(1)
    step_level["prior_last_step"] = gb[step_col].shift(1)

    # Exact running median via an incrementally maintained sorted list (bisect.insort).
    # A pure cumulative-sum trick cannot yield an exact median, so this loop is over the
    # much smaller (key, step)-group table (~572k rows here), not the raw transactions.
    medians = np.empty(len(step_level))
    prev_key = object()
    sorted_list: list = []
    keys = step_level[key_col].to_numpy()
    amts_arr = step_level["amounts"].to_numpy()
    for i in range(len(step_level)):
        k = keys[i]
        if k != prev_key:
            sorted_list = []
            prev_key = k
        n = len(sorted_list)
        if n == 0:
            medians[i] = np.nan
        elif n % 2 == 1:
            medians[i] = sorted_list[n // 2]
        else:
            medians[i] = (sorted_list[n // 2 - 1] + sorted_list[n // 2]) / 2.0
        for a in amts_arr[i]:
            bisect.insort(sorted_list, a)
    step_level["prior_median"] = medians

    prior_count = step_level["prior_count"]
    safe_count = prior_count.replace(0, np.nan)
    step_level["prior_mean"] = step_level["prior_sum"] / safe_count
    var = ((step_level["prior_sumsq"] - step_level["prior_sum"] ** 2 / safe_count)
           / (prior_count - 1).replace(0, np.nan))
    step_level["prior_std"] = np.where(prior_count >= 2, np.sqrt(var.clip(lower=0)), np.nan)

    rename_map = {
        "prior_count": f"{prefix}_tx_count",
        "prior_mean": f"{prefix}_mean_amount",
        "prior_median": f"{prefix}_median_amount",
        "prior_std": f"{prefix}_std_amount",
        "prior_max": f"{prefix}_max_amount",
        "prior_sum": f"{prefix}_total_amount",
        "prior_last_step": f"{prefix}_last_step",
    }
    out = step_level[[key_col, step_col] + list(rename_map.keys())].rename(columns=rename_map)
    out[f"{prefix}_tx_count"] = out[f"{prefix}_tx_count"].astype(int)
    return out


def compute_customer_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe customer behavioral features (Phase 3 Section 6) to df.

    Computed over the FULL chronologically-ordered dataset (not reset per split): a
    customer's transaction history from the training period is legitimately available
    when later scoring that same customer in validation/test -- this is not leakage,
    it mirrors what a production system would actually have on hand. Only statistics
    that are *fit* on data (e.g. imputers/scalers/encoders) are restricted to train-only;
    see build_preprocessor().
    """
    stats = _prior_expanding_stats(df, key_col="customer", prefix="customer_prior")
    merged = df.merge(stats, on=["customer", STEP_COL], how="left")

    merged["is_first_customer_tx"] = (merged["customer_prior_tx_count"] == 0).astype(int)
    merged["time_since_customer_last_tx"] = merged[STEP_COL] - merged["customer_prior_last_step"]

    safe_mean = merged["customer_prior_mean_amount"].replace(0, np.nan)
    safe_median = merged["customer_prior_median_amount"].replace(0, np.nan)
    merged["amount_vs_prior_customer_mean"] = merged[AMOUNT_COL] / safe_mean
    merged["amount_vs_prior_customer_median"] = merged[AMOUNT_COL] / safe_median

    return merged.drop(columns=["customer_prior_last_step"])


def compute_merchant_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe, NON-TARGET merchant behavioral features (Phase 3 Section 8).

    Only volume/amount statistics -- explicitly no merchant historical fraud rate
    (that would use the target and is deferred; see Phase 3 Section 8/9).
    """
    stats = _prior_expanding_stats(df, key_col="merchant", prefix="merchant_prior")
    keep = ["merchant", STEP_COL, "merchant_prior_tx_count",
            "merchant_prior_mean_amount", "merchant_prior_median_amount"]
    merged = df.merge(stats[keep], on=["merchant", STEP_COL], how="left")
    return merged


# ---------------------------------------------------------------------------
# C. Feature sets for Phase 4
# ---------------------------------------------------------------------------

FEATURE_SET_A = ["amount", "category", "merchant", "age", "gender", "is_enterprise"]

FEATURE_SET_B = FEATURE_SET_A + [
    "customer_prior_tx_count",
    "customer_prior_mean_amount",
    "customer_prior_median_amount",
    "customer_prior_std_amount",
    "customer_prior_max_amount",
    "customer_prior_total_amount",
    "amount_vs_prior_customer_mean",
    "amount_vs_prior_customer_median",
    "time_since_customer_last_tx",
    "is_first_customer_tx",
]

FEATURE_SET_C = FEATURE_SET_B + [
    "merchant_prior_tx_count",
    "merchant_prior_mean_amount",
    "merchant_prior_median_amount",
]

CATEGORICAL_FEATURES = ["category", "merchant", "age", "gender"]


def numeric_features_for(feature_set: list) -> list:
    return [f for f in feature_set if f not in CATEGORICAL_FEATURES]


def categorical_features_for(feature_set: list) -> list:
    return [f for f in feature_set if f in CATEGORICAL_FEATURES]


# ---------------------------------------------------------------------------
# D. Preprocessing architecture (Phase 3 Sections 10-11)
# ---------------------------------------------------------------------------

def build_preprocessor(feature_set: list, model_family: str = "linear") -> ColumnTransformer:
    """Build a ColumnTransformer for the given feature set and model family.

    model_family='linear'  -> median-impute + StandardScaler for numeric, most-frequent-
                               impute + OneHotEncoder(handle_unknown='ignore') for categorical.
    model_family='tree'    -> same imputation/encoding, no scaling (unnecessary for trees).

    Imputers/scalers/encoders are only ever .fit() on the training split, because they
    live inside this ColumnTransformer, which the Phase 4 pipeline will fit on X_train
    only -- this is what makes cold-start global fallback values leakage-safe.
    """
    if model_family not in {"linear", "tree"}:
        raise ValueError(f"Unknown model_family: {model_family!r}")

    numeric = numeric_features_for(feature_set)
    categorical = categorical_features_for(feature_set)

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if model_family == "linear":
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])
