"""Data loading, cleaning, and chronological split utilities for the fraud dataset.

Phase 3 prediction-moment principle: a transaction arriving at time `step` may only be
scored using (a) its own fields and (b) information from transactions strictly BEFORE
`step`. No future transactions and no `fraud` labels (current or future) may be used.
This module implements the chronological train/validation/test split that operationalizes
that principle; `features.py` implements the leakage-safe features themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUOTE_STRIPPED_COLS = ["customer", "age", "gender", "zipcodeOri", "merchant", "zipMerchant", "category"]
CONSTANT_COLS = ["zipcodeOri", "zipMerchant"]
TARGET_COL = "fraud"
STEP_COL = "step"

# Chronological split boundaries (inclusive `step` ranges), frozen from Phase 3 Section 3.
# Aligned to full calendar days over the 180-hour window (day = step // 24). Day 7 is a
# partial day (steps 168-179, 12 hours) and is folded into the test period rather than
# left as an undersized standalone block; see the notebook for the full rationale.
TRAIN_STEP_RANGE = (0, 119)    # days 0-4  (5 full days)
VAL_STEP_RANGE = (120, 143)    # day 5     (1 full day)
TEST_STEP_RANGE = (144, 179)   # days 6-7  (1 full day + partial day 7)

SPLIT_ORDER = ["train", "validation", "test"]


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load the raw fraud CSV and strip the literal quote characters from string columns."""
    df = pd.read_csv(path)
    for col in QUOTE_STRIPPED_COLS:
        df[col] = df[col].str.strip("'")
    return df.reset_index(drop=True)


def assign_split(df: pd.DataFrame, step_col: str = STEP_COL) -> pd.Series:
    """Label each row 'train' / 'validation' / 'test' from `step` using the frozen boundaries."""
    split = pd.Series(pd.NA, index=df.index, dtype="object")
    split[df[step_col].between(*TRAIN_STEP_RANGE)] = "train"
    split[df[step_col].between(*VAL_STEP_RANGE)] = "validation"
    split[df[step_col].between(*TEST_STEP_RANGE)] = "test"
    if split.isna().any():
        bad_steps = sorted(df.loc[split.isna(), step_col].unique())
        raise ValueError(
            f"Steps outside the defined split boundaries (update TRAIN/VAL/TEST_STEP_RANGE): {bad_steps}"
        )
    return pd.Categorical(split, categories=SPLIT_ORDER, ordered=True)


def split_summary(df: pd.DataFrame, split_col: str = "split", target_col: str = TARGET_COL) -> pd.DataFrame:
    """Transactions / fraud count / fraud rate per split, in train -> validation -> test order."""
    g = df.groupby(split_col, observed=True)[target_col].agg(transactions="count", fraud="sum")
    g["fraud_rate_pct"] = (g["fraud"] / g["transactions"] * 100).round(3)
    step_range = df.groupby(split_col, observed=True)[STEP_COL].agg(step_min="min", step_max="max")
    return step_range.join(g).reindex(SPLIT_ORDER)


def category_composition_by_split(df: pd.DataFrame, split_col: str = "split",
                                   category_col: str = "category") -> pd.DataFrame:
    """Row share (%) of each category within each split -- for reporting composition drift."""
    tab = pd.crosstab(df[category_col], df[split_col], normalize="columns") * 100
    return tab.reindex(columns=SPLIT_ORDER).round(2)
