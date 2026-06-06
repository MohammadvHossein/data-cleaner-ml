"""Tests for auto_scaler module."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

from data_cleaner.auto_scaler import select_best_scaler


def test_normal_no_outliers_returns_standard():
    np.random.seed(42)
    raw = np.random.normal(50, 10, 100)
    q1, q3 = np.percentile(raw, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    clean = raw[(raw >= lo) & (raw <= hi)]
    s = pd.Series(clean)
    scaler = select_best_scaler(s)
    assert isinstance(scaler, StandardScaler)


def test_has_outliers_returns_robust():
    s = pd.Series(np.random.normal(50, 10, 1000))
    s.iloc[0] = 500
    scaler = select_best_scaler(s)
    assert isinstance(scaler, RobustScaler)


def test_bounded_01_returns_minmax():
    s = pd.Series(np.random.uniform(0, 1, 500))
    scaler = select_best_scaler(s)
    assert isinstance(scaler, MinMaxScaler)


def test_sparse_returns_maxabs():
    np.random.seed(42)
    zeros = np.zeros(500)
    vals = np.random.uniform(0, 10, 500)
    s = pd.Series(np.concatenate([zeros, vals]))
    scaler = select_best_scaler(s)
    assert isinstance(scaler, MaxAbsScaler)


def test_small_sample_defaults_to_standard():
    s = pd.Series([1, 2, 3])
    scaler = select_best_scaler(s)
    assert isinstance(scaler, StandardScaler)


def test_all_nan_handling():
    s = pd.Series([np.nan, np.nan, np.nan])
    scaler = select_best_scaler(s)
    assert isinstance(scaler, StandardScaler)
