import numpy as np
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


def select_best_scaler(series):
    series = series.dropna()
    if len(series) < 10:
        return StandardScaler()

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    has_outliers = (series.min() < lower_bound) or (series.max() > upper_bound)

    is_normal = False
    if len(series) < 5000:
        _, p_value = sp_stats.shapiro(series.sample(min(len(series), 500), random_state=42))
        is_normal = p_value > 0.05
    else:
        _, p_value = sp_stats.normaltest(series.sample(1000, random_state=42))
        is_normal = p_value > 0.05

    col_min, col_max = series.min(), series.max()
    data_range = col_max - col_min
    is_bounded_01 = (data_range > 0) and (col_min >= 0) and (col_max <= 1)
    is_sparse = (series == 0).mean() > 0.4

    if is_normal and not has_outliers:
        return StandardScaler()
    if has_outliers:
        return RobustScaler()
    if is_bounded_01:
        return MinMaxScaler()
    if is_sparse:
        return MaxAbsScaler()

    return StandardScaler()
