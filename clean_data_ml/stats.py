"""Statistical test suite for data analysis.

Provides standalone hypothesis-testing functions (z-test, t-test, chi-square,
ANOVA, AB testing, etc.) and a ``StatisticalTestSuite`` class that integrates
with ``DataCleaner``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)

# -- Type aliases -----------------------------------------------------------

TestResult = Dict[str, Any]


# -- Normality ---------------------------------------------------------------


def normality_test(series: pd.Series, method: str = "shapiro") -> TestResult:
    """Test whether a series is normally distributed.

    Parameters
    ----------
    series : pd.Series
        Data to test.
    method : {"shapiro", "dagostino", "anderson"}
        Normality test to use. Default is ``"shapiro"``.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``is_normal`` (p > 0.05), ``method``.
        Returns ``None`` values with a ``note`` if sample size is insufficient.
    """
    series = series.dropna()
    if len(series) < 4:
        return {"statistic": None, "p_value": None, "is_normal": None, "note": "Insufficient samples (n<4)"}

    if method == "shapiro":
        if len(series) > 5000:
            stat, p = sp_stats.shapiro(series.sample(5000, random_state=42))
        else:
            stat, p = sp_stats.shapiro(series)
    elif method == "dagostino":
        stat, p = sp_stats.normaltest(series)
    elif method == "anderson":
        result = sp_stats.anderson(series, dist="norm")
        stat = result.statistic
        p = result.significance_level[2]
    else:
        raise ValueError(f"Unknown method: {method}")

    return {"statistic": round(float(stat), 4), "p_value": float(p), "is_normal": bool(p > 0.05), "method": method}


# -- Correlation -------------------------------------------------------------


def correlation_test(x: pd.Series, y: pd.Series, method: str = "pearson") -> TestResult:
    """Compute correlation between two series.

    Parameters
    ----------
    x : pd.Series
        First variable.
    y : pd.Series
        Second variable.
    method : {"pearson", "spearman", "kendall"}
        Correlation type. Default ``"pearson"``.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``method``.
    """
    x_arr, y_arr = np.array(x), np.array(y)
    common = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr, y_arr = x_arr[common], y_arr[common]
    if len(x_arr) < 3:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}

    if method == "pearson":
        stat, p = sp_stats.pearsonr(x_arr, y_arr)
    elif method == "spearman":
        stat, p = sp_stats.spearmanr(x_arr, y_arr)
    elif method == "kendall":
        stat, p = sp_stats.kendalltau(x_arr, y_arr)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {"statistic": round(float(stat), 4), "p_value": float(p), "method": method}


# -- Distribution comparison -------------------------------------------------


def ks_test(series_a: pd.Series, series_b: pd.Series) -> TestResult:
    """Two-sample Kolmogorov-Smirnov test.

    Tests whether two samples come from the same distribution.

    Parameters
    ----------
    series_a : pd.Series
        First sample.
    series_b : pd.Series
        Second sample.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``same_distribution`` (p > 0.05).
    """
    a, b = series_a.dropna(), series_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    stat, p = sp_stats.ks_2samp(a, b)
    return {"statistic": round(float(stat), 4), "p_value": float(p), "same_distribution": bool(p > 0.05)}


# -- Chi-square --------------------------------------------------------------


def chi_square_test(series_a: pd.Series, series_b: pd.Series) -> TestResult:
    """Chi-square test of independence for two categorical series.

    Parameters
    ----------
    series_a : pd.Series
        First categorical variable.
    series_b : pd.Series
        Second categorical variable.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``dof``, ``independent`` (p > 0.05).
    """
    table = pd.crosstab(series_a, series_b)
    if table.size == 0:
        return {"statistic": None, "p_value": None, "note": "Empty contingency table"}
    stat, p, dof, expected = sp_stats.chi2_contingency(table)
    return {"statistic": round(float(stat), 4), "p_value": float(p), "dof": int(dof), "independent": bool(p > 0.05)}


# -- Variance equality -------------------------------------------------------


def variance_test(series_a: pd.Series, series_b: pd.Series, method: str = "levene") -> TestResult:
    """Test whether two samples have equal variance.

    Parameters
    ----------
    series_a : pd.Series
        First sample.
    series_b : pd.Series
        Second sample.
    method : {"levene", "bartlett", "fligner"}
        Variance test to use. Default ``"levene"``.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``equal_variance``, ``method``.
    """
    a, b = series_a.dropna(), series_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}

    if method == "levene":
        stat, p = sp_stats.levene(a, b)
    elif method == "bartlett":
        stat, p = sp_stats.bartlett(a, b)
    elif method == "fligner":
        stat, p = sp_stats.fligner(a, b)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {"statistic": round(float(stat), 4), "p_value": float(p), "equal_variance": bool(p > 0.05), "method": method}


# -- ANOVA -------------------------------------------------------------------


def anova_one_way(*groups: pd.Series) -> TestResult:
    """One-way ANOVA test.

    Tests whether the means of two or more groups are significantly different.

    Parameters
    ----------
    *groups : pd.Series
        Two or more samples.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``significant`` (p < 0.05).
    """
    cleaned = [g.dropna() for g in groups]
    cleaned = [g for g in cleaned if len(g) > 1]
    if len(cleaned) < 2:
        return {"statistic": None, "p_value": None, "note": "Need at least 2 groups with >1 sample"}
    stat, p = sp_stats.f_oneway(*cleaned)
    return {"statistic": round(float(stat), 4), "p_value": float(p), "significant": bool(p < 0.05)}


# -- Z-tests -----------------------------------------------------------------


def z_test_one_sample(series: pd.Series, pop_mean: float, pop_std: Optional[float] = None) -> TestResult:
    """One-sample z-test for population mean.

    Parameters
    ----------
    series : pd.Series
        Sample data.
    pop_mean : float
        Hypothesized population mean.
    pop_std : float, optional
        Known population standard deviation. If None, uses sample std.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``method``.
    """
    series = series.dropna()
    n = len(series)
    if n < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    sample_mean = float(series.mean())
    se = pop_std / np.sqrt(n) if pop_std is not None else float(series.std(ddof=1)) / np.sqrt(n)
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}
    z = (sample_mean - pop_mean) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return {"statistic": round(float(z), 4), "p_value": float(p), "method": "one-sample z-test"}


def z_test_two_sample(
    series_a: pd.Series,
    series_b: pd.Series,
    pop_std_a: Optional[float] = None,
    pop_std_b: Optional[float] = None,
) -> TestResult:
    """Two-sample z-test for difference in means.

    Parameters
    ----------
    series_a : pd.Series
        First sample.
    series_b : pd.Series
        Second sample.
    pop_std_a : float, optional
        Known population std for group A.
    pop_std_b : float, optional
        Known population std for group B.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``method``.
    """
    a, b = series_a.dropna(), series_b.dropna()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    m1, m2 = float(a.mean()), float(b.mean())
    if pop_std_a is not None and pop_std_b is not None:
        se = np.sqrt(pop_std_a**2 / n1 + pop_std_b**2 / n2)
    else:
        se = np.sqrt(float(a.var(ddof=1)) / n1 + float(b.var(ddof=1)) / n2)
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}
    z = (m1 - m2) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return {"statistic": round(float(z), 4), "p_value": float(p), "method": "two-sample z-test"}


def z_test_proportion(successes: int, n: int, p_pop: float, alternative: str = "two-sided") -> TestResult:
    """One-sample proportion z-test.

    Parameters
    ----------
    successes : int
        Number of successes observed.
    n : int
        Total number of trials.
    p_pop : float
        Hypothesized population proportion.
    alternative : {"two-sided", "greater", "less"}
        Direction of the test. Default ``"two-sided"``.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``p_hat``, ``method``.
    """
    if n == 0:
        return {"statistic": None, "p_value": None, "note": "Zero trials"}
    p_hat = successes / n
    se = np.sqrt(p_pop * (1 - p_pop) / n)
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}
    z = (p_hat - p_pop) / se
    if alternative == "two-sided":
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - sp_stats.norm.cdf(z)
    elif alternative == "less":
        p = sp_stats.norm.cdf(z)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return {
        "statistic": round(float(z), 4), "p_value": float(p),
        "p_hat": round(float(p_hat), 4), "method": "proportion z-test",
    }


def z_test_two_proportion(
    successes_a: int, n_a: int, successes_b: int, n_b: int, alternative: str = "two-sided"
) -> TestResult:
    """Two-sample proportion z-test (pooled).

    Parameters
    ----------
    successes_a : int
        Successes in group A.
    n_a : int
        Total trials in group A.
    successes_b : int
        Successes in group B.
    n_b : int
        Total trials in group B.
    alternative : {"two-sided", "greater", "less"}
        Direction. Default ``"two-sided"``.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``p1``, ``p2``, ``method``.
    """
    if n_a == 0 or n_b == 0:
        return {"statistic": None, "p_value": None, "note": "Zero trials in one group"}
    p1, p2 = successes_a / n_a, successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}
    z = (p1 - p2) / se
    if alternative == "two-sided":
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p = 1 - sp_stats.norm.cdf(z)
    elif alternative == "less":
        p = sp_stats.norm.cdf(z)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return {
        "statistic": round(float(z), 4),
        "p_value": float(p),
        "p1": round(float(p1), 4),
        "p2": round(float(p2), 4),
        "method": "two-proportion z-test",
    }


# -- T-tests -----------------------------------------------------------------


def t_test_one_sample(series: pd.Series, pop_mean: float = 0) -> TestResult:
    """One-sample t-test.

    Parameters
    ----------
    series : pd.Series
        Sample data.
    pop_mean : float
        Hypothesized population mean (default 0).

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``dof``, ``method``.
    """
    series = series.dropna()
    if len(series) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    stat, p = sp_stats.ttest_1samp(series, pop_mean)
    return {
        "statistic": round(float(stat), 4), "p_value": float(p),
        "dof": len(series) - 1, "method": "one-sample t-test",
    }


def t_test_independent(series_a: pd.Series, series_b: pd.Series, equal_var: bool = True) -> TestResult:
    """Independent two-sample t-test.

    Parameters
    ----------
    series_a : pd.Series
        First sample.
    series_b : pd.Series
        Second sample.
    equal_var : bool
        If True (default), assumes equal variance (Student's t).
        If False, uses Welch's t-test.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``dof``, ``equal_var``, ``method``.
    """
    a, b = series_a.dropna(), series_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    stat, p = sp_stats.ttest_ind(a, b, equal_var=equal_var)
    return {
        "statistic": round(float(stat), 4),
        "p_value": float(p),
        "dof": len(a) + len(b) - 2,
        "equal_var": equal_var,
        "method": "independent t-test",
    }


def t_test_paired(series_a: pd.Series, series_b: pd.Series) -> TestResult:
    """Paired (dependent) two-sample t-test.

    Parameters
    ----------
    series_a : pd.Series
        First measurement.
    series_b : pd.Series
        Second measurement (same subjects).

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``dof``, ``method``.
    """
    a, b = np.array(series_a.dropna()), np.array(series_b.dropna())
    common = ~(np.isnan(a) | np.isnan(b))
    a, b = a[common], b[common]
    if len(a) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    stat, p = sp_stats.ttest_rel(a, b)
    return {"statistic": round(float(stat), 4), "p_value": float(p), "dof": len(a) - 1, "method": "paired t-test"}


# -- AB testing --------------------------------------------------------------


def ab_test(
    control_data: Union[pd.Series, int, float],
    treatment_data: Union[pd.Series, int, float],
    metric_type: str = "mean",
    alpha: float = 0.05,
) -> TestResult:
    """Dispatch AB test by metric type.

    Parameters
    ----------
    control_data : pd.Series or int/float
        Control group data or number of successes.
    treatment_data : pd.Series or int/float
        Treatment group data or total trials.
    metric_type : {"mean", "proportion"}
        Type of metric being tested. Default ``"mean"``.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict
        Result from ``ab_test_mean`` or ``ab_test_proportion``.
    """
    if isinstance(control_data, (int, float, np.integer)) and isinstance(treatment_data, (int, float, np.integer)):
        return ab_test_proportion(control_data, treatment_data, alpha=alpha)
    if metric_type == "mean":
        return ab_test_mean(control_data, treatment_data, alpha=alpha)
    elif metric_type == "proportion":
        return ab_test_proportion(control_data, treatment_data, alpha=alpha)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def ab_test_mean(
    control: pd.Series, treatment: pd.Series, alpha: float = 0.05
) -> TestResult:
    """AB test for difference in means (z-test based).

    Parameters
    ----------
    control : pd.Series
        Control group values.
    treatment : pd.Series
        Treatment group values.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``control_mean``, ``treatment_mean``,
        ``lift_pct``, ``ci``, ``significant``, ``n_control``, ``n_treatment``.
    """
    c, t = np.array(control.dropna()), np.array(treatment.dropna())
    n1, n2 = len(c), len(t)
    if n1 < 2 or n2 < 2:
        return {"note": "Insufficient samples", "significant": None}  # type: ignore[return-value]
    m1, m2 = float(c.mean()), float(t.mean())
    v1, v2 = float(c.var(ddof=1)), float(t.var(ddof=1))
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}  # type: ignore[return-value]
    z = (m2 - m1) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    lift = (m2 - m1) / m1 * 100 if m1 != 0 else None
    moe = float(sp_stats.norm.ppf(1 - alpha / 2)) * se
    ci = (m2 - m1 - moe, m2 - m1 + moe)
    return {
        "statistic": round(float(z), 4),
        "p_value": float(p),
        "control_mean": round(m1, 4),
        "treatment_mean": round(m2, 4),
        "lift_pct": round(lift, 2) if lift is not None else None,
        "ci": (round(float(ci[0]), 4), round(float(ci[1]), 4)),
        "significant": bool(p < alpha),
        "alpha": alpha,
        "n_control": n1,
        "n_treatment": n2,
        "method": "AB test (mean, z-test)",
    }


def ab_test_proportion(
    control: Union[pd.Series, int, float],
    treatment: Union[pd.Series, int, float],
    alpha: float = 0.05,
) -> TestResult:
    """AB test for difference in proportions.

    Parameters
    ----------
    control : pd.Series (0/1) or int
        Control group. If Series, 0/1 values. If int, number of successes.
    treatment : pd.Series (0/1) or int
        Treatment group. If Series, 0/1 values. If int, total trials.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``control_rate``, ``treatment_rate``,
        ``lift_pct``, ``ci``, ``significant``, ``n_control``, ``n_treatment``.
    """
    if isinstance(control, (int, float, np.integer)):
        return {
            "note": "Single proportion - use z_test_proportion or provide both groups",
            "method": "AB test (proportion)",
        }  # type: ignore[return-value]

    control_arr = np.array(control)
    treatment_arr = np.array(treatment)
    if control_arr.ndim == 1 and treatment_arr.ndim == 1:
        c_success = int(control_arr.sum())
        c_total = len(control_arr)
        t_success = int(treatment_arr.sum())
        t_total = len(treatment_arr)
    else:
        return {"note": "Expected 1D arrays of 0/1 values"}  # type: ignore[return-value]

    return _ab_proportion_test(c_success, c_total, t_success, t_total, alpha)


def _ab_proportion_test(
    c_success: int, c_total: int, t_success: int, t_total: int, alpha: float = 0.05
) -> TestResult:
    """Internal pooled two-proportion z-test for AB testing."""
    p1, p2 = c_success / c_total, t_success / t_total
    p_pool = (c_success + t_success) / (c_total + t_total)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / c_total + 1 / t_total))
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}  # type: ignore[return-value]
    z = (p2 - p1) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    lift = (p2 - p1) / p1 * 100 if p1 != 0 else None
    moe = float(sp_stats.norm.ppf(1 - alpha / 2)) * se
    ci = (p2 - p1 - moe, p2 - p1 + moe)
    return {
        "statistic": round(float(z), 4),
        "p_value": float(p),
        "control_rate": round(float(p1), 4),
        "treatment_rate": round(float(p2), 4),
        "lift_pct": round(lift, 2) if lift is not None else None,
        "ci": (round(float(ci[0]), 4), round(float(ci[1]), 4)),
        "significant": bool(p < alpha),
        "alpha": alpha,
        "n_control": c_total,
        "n_treatment": t_total,
        "method": "AB test (proportion, z-test)",
    }


# -- Mutual Information ------------------------------------------------------


def mutual_information(X: Union[pd.DataFrame, np.ndarray], y: pd.Series) -> Dict[str, float]:
    """Compute mutual information between features and target.

    Automatically selects ``mutual_info_classif`` (<20 unique values)
    or ``mutual_info_regression``.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    dict
        Column name -> MI score.
    """
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include="number").dropna()
    else:
        X_num = pd.DataFrame(X).select_dtypes(include="number").dropna()
    y_clean = y.loc[X_num.index]
    if y_clean.nunique() < 20:
        mi = mutual_info_classif(X_num, y_clean, random_state=42)
    else:
        mi = mutual_info_regression(X_num, y_clean, random_state=42)
    return dict(zip(X_num.columns, [round(float(v), 4) for v in mi]))


# -- Test Suite --------------------------------------------------------------


class StatisticalTestSuite:
    """High-level test suite that wraps ``DataCleaner`` data.

    Runs statistical tests on the loaded DataFrame and stores results
    for later inspection via ``.summary()``.

    Parameters
    ----------
    dc : DataCleaner, optional
        A DataCleaner instance with loaded data.
    """

    def __init__(self, dc: Optional["DataCleaner"] = None) -> None:  # noqa: F821
        self.dc = dc
        self.results: Dict[str, Any] = {}

    def test_normality(self, columns: Optional[List[str]] = None, method: str = "shapiro") -> Dict[str, TestResult]:
        """Test normality of specified (or all) numeric columns.

        Parameters
        ----------
        columns : list of str, optional
            Columns to test. If None, all numeric columns are tested.
        method : {"shapiro", "dagostino", "anderson"}
            Normality test method.

        Returns
        -------
        dict
            Column name -> normality_test result.
        """
        self._require_data()
        df = self.dc.df  # type: ignore[union-attr]
        if columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()
        results = {}
        for col in columns:
            if col not in df.columns:
                continue
            results[col] = normality_test(df[col], method=method)
        self.results["normality"] = results
        return results

    def test_correlations(
        self, target_col: Optional[str] = None, method: str = "pearson"
    ) -> Union[Dict[str, TestResult], pd.DataFrame]:
        """Test correlations between numeric columns.

        Parameters
        ----------
        target_col : str, optional
            If set, returns correlations of all other columns with this target.
        method : {"pearson", "spearman", "kendall"}

        Returns
        -------
        dict or pd.DataFrame
            Column->result dict if target_col given, else correlation matrix.
        """
        self._require_data()
        df = self.dc.df.select_dtypes(include="number")  # type: ignore[union-attr]
        if target_col:
            if target_col not in df.columns:
                raise ValueError(f"Column '{target_col}' not found")
            results = {}
            for col in df.columns:
                if col == target_col:
                    continue
                results[col] = correlation_test(df[col], df[target_col], method=method)
            self.results[f"correlation_{target_col}"] = results
            return results
        corr_matrix = df.corr(method=method if method in ("pearson", "spearman", "kendall") else "pearson")
        self.results["correlation_matrix"] = corr_matrix
        return corr_matrix

    def test_chi_square(self, col_a: str, col_b: str) -> TestResult:
        """Chi-square test of independence between two categorical columns.

        Parameters
        ----------
        col_a : str
            First column name.
        col_b : str
            Second column name.

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        result = chi_square_test(self.dc.df[col_a], self.dc.df[col_b])  # type: ignore[index]
        self.results[f"chisquare_{col_a}_{col_b}"] = result
        return result

    def test_anova(self, target_col: str, group_col: str) -> TestResult:
        """One-way ANOVA of target_col grouped by group_col.

        Parameters
        ----------
        target_col : str
            Numeric column.
        group_col : str
            Categorical grouping column.

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        groups = [g for _, g in self.dc.df.groupby(group_col)[target_col]]  # type: ignore[union-attr]
        result = anova_one_way(*groups)
        self.results[f"anova_{target_col}_by_{group_col}"] = result
        return result

    def test_z_one_sample(self, column: str, pop_mean: float, pop_std: Optional[float] = None) -> TestResult:
        """One-sample z-test on a column.

        Parameters
        ----------
        column : str
            Column name.
        pop_mean : float
            Hypothesized mean.
        pop_std : float, optional
            Known population std.

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        result = z_test_one_sample(self.dc.df[column], pop_mean, pop_std)  # type: ignore[index]
        self.results[f"z_onesample_{column}"] = result
        return result

    def test_z_two_sample(self, col_a: str, col_b: str) -> TestResult:
        """Two-sample z-test between two columns.

        Parameters
        ----------
        col_a : str
            First column.
        col_b : str
            Second column.

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        result = z_test_two_sample(self.dc.df[col_a], self.dc.df[col_b])  # type: ignore[index]
        self.results[f"z_twosample_{col_a}_{col_b}"] = result
        return result

    def test_z_proportion(self, column: str, success_value: Any, p_pop: float) -> TestResult:
        """One-sample proportion test on a column.

        Parameters
        ----------
        column : str
            Column name (should contain 0/1 or categorical values).
        success_value : any
            Value treated as "success".
        p_pop : float
            Hypothesized population proportion.

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        series = self.dc.df[column]  # type: ignore[index]
        successes = int((series == success_value).sum())
        n = len(series)
        result = z_test_proportion(successes, n, p_pop)
        self.results[f"z_proportion_{column}"] = result
        return result

    def test_z_two_proportion(self, col_a: str, col_b: str, success_value: Any = 1) -> TestResult:
        """Two-sample proportion test between two columns.

        Parameters
        ----------
        col_a : str
            First column.
        col_b : str
            Second column.
        success_value : any
            Value treated as "success" (default 1).

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        df = self.dc.df  # type: ignore[union-attr]
        s_a = int((df[col_a] == success_value).sum())
        s_b = int((df[col_b] == success_value).sum())
        result = z_test_two_proportion(s_a, len(df), s_b, len(df))
        self.results[f"z_two_proportion_{col_a}_{col_b}"] = result
        return result

    def test_t_one_sample(self, column: str, pop_mean: float = 0) -> TestResult:
        """One-sample t-test on a column.

        Parameters
        ----------
        column : str
            Column name.
        pop_mean : float
            Hypothesized mean (default 0).

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        result = t_test_one_sample(self.dc.df[column], pop_mean)  # type: ignore[index]
        self.results[f"t_onesample_{column}"] = result
        return result

    def test_t_independent(self, col_a: str, col_b: str, equal_var: bool = True) -> TestResult:
        """Independent t-test between two columns.

        Parameters
        ----------
        col_a : str
            First column.
        col_b : str
            Second column.
        equal_var : bool
            Assume equal variance (default True).

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        result = t_test_independent(self.dc.df[col_a], self.dc.df[col_b], equal_var=equal_var)  # type: ignore[index]
        self.results[f"t_independent_{col_a}_{col_b}"] = result
        return result

    def test_t_paired(self, col_a: str, col_b: str) -> TestResult:
        """Paired t-test between two columns.

        Parameters
        ----------
        col_a : str
            First column (before).
        col_b : str
            Second column (after).

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        result = t_test_paired(self.dc.df[col_a], self.dc.df[col_b])  # type: ignore[index]
        self.results[f"t_paired_{col_a}_{col_b}"] = result
        return result

    def test_ab(
        self, control_col: str, treatment_col: str, metric_type: str = "mean", alpha: float = 0.05
    ) -> TestResult:
        """AB test comparing two columns.

        Parameters
        ----------
        control_col : str
            Control column name.
        treatment_col : str
            Treatment column name.
        metric_type : {"mean", "proportion"}
            Type of metric (default "mean").
        alpha : float
            Significance level (default 0.05).

        Returns
        -------
        dict
            Test result with significance, lift, CI.
        """
        self._require_data()
        result = ab_test(
            self.dc.df[control_col], self.dc.df[treatment_col],  # type: ignore[index]
            metric_type=metric_type, alpha=alpha,
        )
        self.results["ab_test"] = result
        return result

    def test_ab_by_group(
        self,
        metric_col: str,
        group_col: str,
        control_value: Any,
        treatment_value: Any,
        metric_type: str = "mean",
        alpha: float = 0.05,
    ) -> TestResult:
        """AB test by splitting a group column into control/treatment.

        Parameters
        ----------
        metric_col : str
            Column with the metric values.
        group_col : str
            Column identifying control vs treatment.
        control_value : any
            Value in group_col identifying the control group.
        treatment_value : any
            Value in group_col identifying the treatment group.
        metric_type : {"mean", "proportion"}
            Type of metric (default "mean").
        alpha : float
            Significance level (default 0.05).

        Returns
        -------
        dict
            Test result.
        """
        self._require_data()
        df = self.dc.df  # type: ignore[union-attr]
        control = df[df[group_col] == control_value][metric_col]
        treatment = df[df[group_col] == treatment_value][metric_col]

        if metric_type == "mean":
            result = ab_test_mean(control, treatment, alpha=alpha)
        elif metric_type == "proportion":
            result = _ab_proportion_test(
                int(control.sum()), int(len(control)),
                int(treatment.sum()), int(len(treatment)),
                alpha,
            )
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        result["control_group"] = control_value
        result["treatment_group"] = treatment_value
        self.results["ab_test"] = result
        return result

    def test_mutual_info(self, target_col: Optional[str] = None) -> Dict[str, float]:
        """Compute mutual information for all features against a target.

        Parameters
        ----------
        target_col : str, optional
            Target column. Uses ``dc.target_col`` if not given.

        Returns
        -------
        dict
            Column -> MI score.
        """
        self._require_data()
        df = self.dc.df  # type: ignore[union-attr]
        if target_col is None:
            target_col = self.dc.target_col  # type: ignore[union-attr]
        if target_col is None:
            raise ValueError("No target column specified")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        result = mutual_information(X, y)
        self.results["mutual_info"] = result
        return result

    def summary(self) -> str:
        """Return a formatted summary of all test results.

        Includes normality results, mutual information, z/t-tests,
        and AB test results with significance labels.

        Returns
        -------
        str
            Multi-line string with test summaries.
        """
        lines: List[str] = []

        if "normality" in self.results:
            lines.append("=== Normality Tests ===")
            for col, res in self.results["normality"].items():
                status = "NORMAL" if res.get("is_normal") else "NOT NORMAL"
                p = res.get("p_value")
                p_str = f" (p={p:.4f})" if p is not None else ""
                lines.append(f"  {col}: {status}{p_str}")

        if "mutual_info" in self.results:
            lines.append("\n=== Mutual Information ===")
            for col, mi in sorted(self.results["mutual_info"].items(), key=lambda x: -x[1]):
                lines.append(f"  {col}: {mi:.4f}")

        for key, res in self.results.items():
            if key.startswith("z_") or key.startswith("t_"):
                label = key[0].upper() + "-" + key[2:].replace("_", " ")
                p = res.get("p_value")
                if p is not None:
                    sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
                    lines.append(f"\n=== {label} ===  statistic={res.get('statistic')}, p={p:.4f} ({sig})")
                else:
                    lines.append(f"\n=== {label} ===  {res.get('note', 'N/A')}")

        if "ab_test" in self.results:
            res = self.results["ab_test"]
            lines.append("\n=== AB Test ===")
            if res.get("significant") is not None:
                lines.append(f"  Significant: {'YES' if res['significant'] else 'NO'} (p={res['p_value']:.4f})")
                lines.append(f"  Lift: {res.get('lift_pct')}%")
                ci = res.get("ci")
                if ci:
                    lines.append(f"  CI: ({ci[0]:.4f}, {ci[1]:.4f})")
                ctrl = res.get("control_mean") or res.get("control_rate")
                trt = res.get("treatment_mean") or res.get("treatment_rate")
                lines.append(f"  Control: {ctrl} (n={res.get('n_control')})")
                lines.append(f"  Treatment: {trt} (n={res.get('n_treatment')})")
            else:
                lines.append(f"  {res.get('note', 'N/A')}")

        return "\n".join(lines)

    def _require_data(self) -> None:
        """Ensure a DataCleaner with loaded data is available."""
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
