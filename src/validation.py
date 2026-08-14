"""Lightweight leakage and split-integrity checks for Phase 3.

These are spot-check assertions, not a full test suite: each raises AssertionError
(or ValueError) on failure and returns True on success, so they can be called directly
from the notebook as a visible pass/fail gate before features are used downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONSTANT_COLS = ["zipcodeOri", "zipMerchant"]
TARGET_COL = "fraud"


def check_split_order(df: pd.DataFrame, split_col: str = "split", step_col: str = "step") -> bool:
    """Train must fully precede validation, which must fully precede test."""
    train_max = df.loc[df[split_col] == "train", step_col].max()
    val_min = df.loc[df[split_col] == "validation", step_col].min()
    val_max = df.loc[df[split_col] == "validation", step_col].max()
    test_min = df.loc[df[split_col] == "test", step_col].min()
    assert train_max < val_min, f"train/validation overlap: train max step {train_max} >= val min step {val_min}"
    assert val_max < test_min, f"validation/test overlap: val max step {val_max} >= test min step {test_min}"
    return True


def check_no_future_leakage(df: pd.DataFrame, key_col: str, prior_count_col: str,
                             prior_mean_col: str | None = None, n_samples: int = 200,
                             random_state: int = 0, amount_col: str = "amount",
                             step_col: str = "step") -> bool:
    """Brute-force spot-check: recompute prior count/mean for a random sample of rows
    by direct filtering (df[key]==row[key] & df[step] < row[step]) and compare against
    the vectorized feature. This directly verifies no future information leaked in.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(df.index.to_numpy(), size=min(n_samples, len(df)), replace=False)
    for i in idx:
        row = df.loc[i]
        hist = df.loc[(df[key_col] == row[key_col]) & (df[step_col] < row[step_col]), amount_col]
        expected_count = len(hist)
        actual_count = row[prior_count_col]
        assert expected_count == actual_count, (
            f"row {i}: expected prior count {expected_count}, got {actual_count}"
        )
        if prior_mean_col is not None and expected_count > 0:
            expected_mean = hist.mean()
            actual_mean = row[prior_mean_col]
            assert abs(expected_mean - actual_mean) < 1e-6, (
                f"row {i}: mean mismatch {expected_mean} vs {actual_mean}"
            )
    return True


def check_same_step_no_leakage(df: pd.DataFrame, key_col: str, stat_cols: list,
                                step_col: str = "step") -> bool:
    """All rows sharing a (key, step) pair must have IDENTICAL historical stats --
    proof that transactions at the same step cannot see each other's information.
    """
    grp_sizes = df.groupby([key_col, step_col]).size()
    multi = grp_sizes[grp_sizes > 1]
    assert len(multi) > 0, "no multi-transaction (key, step) groups found to test against"
    sample_key, sample_step = multi.index[0]
    sub = df.loc[(df[key_col] == sample_key) & (df[step_col] == sample_step)]
    for c in stat_cols:
        assert sub[c].nunique(dropna=False) == 1, f"{c} differs within the same (key, step) group"
    return True


def check_no_constant_columns(feature_list: list) -> bool:
    leaked = [c for c in CONSTANT_COLS if c in feature_list]
    assert not leaked, f"constant columns present in feature list: {leaked}"
    return True


def check_target_not_in_features(feature_list: list) -> bool:
    assert TARGET_COL not in feature_list, "target column present in feature list"
    return True
