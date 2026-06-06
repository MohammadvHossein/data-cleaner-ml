"""Automatic scaler selection for numeric columns.

Tests each column for normality, outliers, bounds, and sparsity
to pick the most appropriate sklearn scaler.
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

Scaler = Union[StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler]


def select_best_scaler(series: pd.Series) -> Scaler:
    """Select the optimal scaler for a numeric column based on its distribution.

    Tests normality (Shapiro-Wilk / D'Agostino), outlier presence (IQR),
    value bounds, and sparsity, then returns the best-fit sklearn scaler.

    Parameters
    ----------
    series : pd.Series
        Numeric column with potential null values.

    Returns
    -------
    StandardScaler, RobustScaler, MinMaxScaler, or MaxAbsScaler
        The selected scaler instance (unfitted).

    Selection logic
    ---------------
    - Normal + no outliers  : StandardScaler
    - Has outliers          : RobustScaler
    - Bounded in [0, 1]     : MinMaxScaler
    - Sparse (>40% zeros)   : MaxAbsScaler
    - Default               : StandardScaler
    """
    series = series.dropna()
    if len(series) < 10:
        return StandardScaler()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    has_outliers = bool((series.min() < lower) or (series.max() > upper))

    is_normal = False
    if len(series) < 5000:
        _, p_value = sp_stats.shapiro(series.sample(min(len(series), 500), random_state=42))
        is_normal = bool(p_value > 0.05)
    else:
        _, p_value = sp_stats.normaltest(series.sample(1000, random_state=42))
        is_normal = bool(p_value > 0.05)

    col_min, col_max = float(series.min()), float(series.max())
    data_range = col_max - col_min
    is_bounded_01 = bool(data_range > 0 and col_min >= 0 and col_max <= 1)
    is_sparse = bool((series == 0).mean() > 0.4)

    if is_sparse:
        return MaxAbsScaler()
    if is_normal and not has_outliers:
        return StandardScaler()
    if has_outliers:
        return RobustScaler()
    if is_bounded_01:
        return MinMaxScaler()

    return StandardScaler()
