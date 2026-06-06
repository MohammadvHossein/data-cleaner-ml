import logging
import numpy as np
import pandas as pd

from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)


def normality_test(series, method="shapiro"):
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
    return {"statistic": round(stat, 4), "p_value": p, "is_normal": p > 0.05, "method": method}


def correlation_test(x, y, method="pearson"):
    x, y = np.array(x.dropna()), np.array(y.dropna())
    common = ~(np.isnan(x) | np.isnan(y))
    x, y = x[common], y[common]
    if len(x) < 3:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    if method == "pearson":
        stat, p = sp_stats.pearsonr(x, y)
    elif method == "spearman":
        stat, p = sp_stats.spearmanr(x, y)
    elif method == "kendall":
        stat, p = sp_stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    return {"statistic": round(stat, 4), "p_value": p, "method": method}


def ks_test(series_a, series_b):
    a, b = series_a.dropna(), series_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    stat, p = sp_stats.ks_2samp(a, b)
    return {"statistic": round(stat, 4), "p_value": p, "same_distribution": p > 0.05}


def chi_square_test(series_a, series_b):
    table = pd.crosstab(series_a, series_b)
    if table.size == 0:
        return {"statistic": None, "p_value": None, "note": "Empty contingency table"}
    stat, p, dof, expected = sp_stats.chi2_contingency(table)
    return {"statistic": round(stat, 4), "p_value": p, "dof": dof, "independent": p > 0.05}


def variance_test(series_a, series_b, method="levene"):
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
    return {"statistic": round(stat, 4), "p_value": p, "equal_variance": p > 0.05, "method": method}


def anova_one_way(*groups):
    groups = [g.dropna() for g in groups]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return {"statistic": None, "p_value": None, "note": "Need at least 2 groups with >1 sample"}
    stat, p = sp_stats.f_oneway(*groups)
    return {"statistic": round(stat, 4), "p_value": p, "significant": p < 0.05}


def z_test_one_sample(series, pop_mean, pop_std=None):
    series = series.dropna()
    n = len(series)
    if n < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    sample_mean = series.mean()
    if pop_std is not None:
        se = pop_std / np.sqrt(n)
    else:
        se = series.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}
    z = (sample_mean - pop_mean) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return {"statistic": round(z, 4), "p_value": p, "method": "one-sample z-test"}


def z_test_two_sample(series_a, series_b, pop_std_a=None, pop_std_b=None):
    a, b = series_a.dropna(), series_b.dropna()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    mean1, mean2 = a.mean(), b.mean()
    if pop_std_a is not None and pop_std_b is not None:
        se = np.sqrt(pop_std_a**2 / n1 + pop_std_b**2 / n2)
    else:
        se = np.sqrt(a.var(ddof=1) / n1 + b.var(ddof=1) / n2)
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}
    z = (mean1 - mean2) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return {"statistic": round(z, 4), "p_value": p, "method": "two-sample z-test"}


def z_test_proportion(successes, n, p_pop, alternative="two-sided"):
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
    return {"statistic": round(z, 4), "p_value": p, "p_hat": round(p_hat, 4), "method": "proportion z-test"}


def z_test_two_proportion(successes_a, n_a, successes_b, n_b, alternative="two-sided"):
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
    return {"statistic": round(z, 4), "p_value": p, "p1": round(p1, 4), "p2": round(p2, 4), "method": "two-proportion z-test"}


def t_test_one_sample(series, pop_mean=0):
    series = series.dropna()
    if len(series) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    stat, p = sp_stats.ttest_1samp(series, pop_mean)
    return {"statistic": round(stat, 4), "p_value": p, "dof": len(series) - 1, "method": "one-sample t-test"}


def t_test_independent(series_a, series_b, equal_var=True):
    a, b = series_a.dropna(), series_b.dropna()
    if len(a) < 2 or len(b) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    stat, p = sp_stats.ttest_ind(a, b, equal_var=equal_var)
    return {"statistic": round(stat, 4), "p_value": p, "dof": len(a) + len(b) - 2, "equal_var": equal_var, "method": "independent t-test"}


def t_test_paired(series_a, series_b):
    a, b = np.array(series_a.dropna()), np.array(series_b.dropna())
    common = ~(np.isnan(a) | np.isnan(b))
    a, b = a[common], b[common]
    if len(a) < 2:
        return {"statistic": None, "p_value": None, "note": "Insufficient samples"}
    stat, p = sp_stats.ttest_rel(a, b)
    return {"statistic": round(stat, 4), "p_value": p, "dof": len(a) - 1, "method": "paired t-test"}


def ab_test(control_data, treatment_data, metric_type="mean", alpha=0.05):
    if isinstance(control_data, (int, float, np.integer)) and isinstance(treatment_data, (int, float, np.integer)):
        return ab_test_proportion(control_data, treatment_data, alpha=alpha)
    if metric_type == "mean":
        return ab_test_mean(control_data, treatment_data, alpha=alpha)
    elif metric_type == "proportion":
        return ab_test_proportion(control_data, treatment_data, alpha=alpha)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")


def ab_test_mean(control, treatment, alpha=0.05):
    c, t = np.array(control.dropna()), np.array(treatment.dropna())
    n1, n2 = len(c), len(t)
    if n1 < 2 or n2 < 2:
        return {"note": "Insufficient samples"}
    mean1, mean2 = c.mean(), t.mean()
    var1, var2 = c.var(ddof=1), t.var(ddof=1)
    se = np.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}
    z = (mean2 - mean1) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    lift = (mean2 - mean1) / mean1 * 100 if mean1 != 0 else None
    ci = (mean2 - mean1) - sp_stats.norm.ppf(1 - alpha / 2) * se, (mean2 - mean1) + sp_stats.norm.ppf(1 - alpha / 2) * se
    return {
        "statistic": round(z, 4), "p_value": p,
        "control_mean": round(mean1, 4), "treatment_mean": round(mean2, 4),
        "lift_pct": round(lift, 2) if lift is not None else None,
        "ci": (round(ci[0], 4), round(ci[1], 4)),
        "significant": p < alpha, "alpha": alpha,
        "n_control": n1, "n_treatment": n2,
        "method": "AB test (mean, z-test)"
    }


def ab_test_proportion(control, treatment, alpha=0.05):
    if isinstance(control, (int, float, np.integer)):
        c_success, c_total = control, treatment
        t_success, t_total = None, None
        return {
            "note": "Single proportion - use z_test_proportion or provide both groups",
            "method": "AB test (proportion)"
        }
    else:
        control = np.array(control)
        treatment = np.array(treatment)
        if control.ndim == 1 and treatment.ndim == 1:
            c_success, c_total = control.sum(), len(control)
            t_success, t_total = treatment.sum(), len(treatment)
        else:
            return {"note": "Expected 1D arrays of 0/1 values"}
        return _ab_proportion_test(c_success, c_total, t_success, t_total, alpha)


def _ab_proportion_test(c_success, c_total, t_success, t_total, alpha=0.05):
    p1, p2 = c_success / c_total, t_success / t_total
    p_pool = (c_success + t_success) / (c_total + t_total)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / c_total + 1 / t_total))
    if se == 0:
        return {"statistic": None, "p_value": None, "note": "Zero standard error"}
    z = (p2 - p1) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    lift = (p2 - p1) / p1 * 100 if p1 != 0 else None
    ci = (p2 - p1) - sp_stats.norm.ppf(1 - alpha / 2) * se, (p2 - p1) + sp_stats.norm.ppf(1 - alpha / 2) * se
    return {
        "statistic": round(z, 4), "p_value": p,
        "control_rate": round(p1, 4), "treatment_rate": round(p2, 4),
        "lift_pct": round(lift, 2) if lift is not None else None,
        "ci": (round(ci[0], 4), round(ci[1], 4)),
        "significant": p < alpha, "alpha": alpha,
        "n_control": c_total, "n_treatment": t_total,
        "method": "AB test (proportion, z-test)"
    }


def mutual_information(X, y, problem_type="auto"):
    if isinstance(X, pd.DataFrame):
        X_num = X.select_dtypes(include="number").dropna()
    else:
        X_num = pd.DataFrame(X).select_dtypes(include="number").dropna()
    y_clean = y.loc[X_num.index]
    if y_clean.nunique() < 20:
        mi = mutual_info_classif(X_num, y_clean, random_state=42)
    else:
        mi = mutual_info_regression(X_num, y_clean, random_state=42)
    return dict(zip(X_num.columns, [round(v, 4) for v in mi]))


class StatisticalTestSuite:
    def __init__(self, dc=None):
        self.dc = dc
        self.results = {}

    def test_normality(self, columns=None, method="shapiro"):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        df = self.dc.df
        if columns is None:
            columns = df.select_dtypes(include="number").columns.tolist()
        results = {}
        for col in columns:
            if col not in df.columns:
                continue
            results[col] = normality_test(df[col], method=method)
        self.results["normality"] = results
        return results

    def test_correlations(self, target_col=None, method="pearson"):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        df = self.dc.df.select_dtypes(include="number")
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
        else:
            corr_matrix = df.corr(method=method if method in ("pearson", "spearman", "kendall") else "pearson")
            self.results["correlation_matrix"] = corr_matrix
            return corr_matrix

    def test_chi_square(self, col_a, col_b):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        result = chi_square_test(self.dc.df[col_a], self.dc.df[col_b])
        self.results[f"chisquare_{col_a}_{col_b}"] = result
        return result

    def test_anova(self, target_col, group_col):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        groups = [g for _, g in self.dc.df.groupby(group_col)[target_col]]
        result = anova_one_way(*groups)
        self.results[f"anova_{target_col}_by_{group_col}"] = result
        return result

    def test_z_one_sample(self, column, pop_mean, pop_std=None):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        result = z_test_one_sample(self.dc.df[column], pop_mean, pop_std)
        self.results[f"z_onesample_{column}"] = result
        return result

    def test_z_two_sample(self, col_a, col_b):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        result = z_test_two_sample(self.dc.df[col_a], self.dc.df[col_b])
        self.results[f"z_twosample_{col_a}_{col_b}"] = result
        return result

    def test_z_proportion(self, column, success_value, p_pop):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        series = self.dc.df[column]
        successes = (series == success_value).sum()
        n = len(series)
        result = z_test_proportion(successes, n, p_pop)
        self.results[f"z_proportion_{column}"] = result
        return result

    def test_z_two_proportion(self, col_a, col_b, success_value=1):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        df = self.dc.df
        s_a = (df[col_a] == success_value).sum()
        s_b = (df[col_b] == success_value).sum()
        result = z_test_two_proportion(s_a, len(df), s_b, len(df))
        self.results[f"z_two_proportion_{col_a}_{col_b}"] = result
        return result

    def test_t_one_sample(self, column, pop_mean=0):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        result = t_test_one_sample(self.dc.df[column], pop_mean)
        self.results[f"t_onesample_{column}"] = result
        return result

    def test_t_independent(self, col_a, col_b, equal_var=True):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        result = t_test_independent(self.dc.df[col_a], self.dc.df[col_b], equal_var=equal_var)
        self.results[f"t_independent_{col_a}_{col_b}"] = result
        return result

    def test_t_paired(self, col_a, col_b):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        result = t_test_paired(self.dc.df[col_a], self.dc.df[col_b])
        self.results[f"t_paired_{col_a}_{col_b}"] = result
        return result

    def test_ab(self, control_col, treatment_col, metric_type="mean", alpha=0.05):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        result = ab_test(self.dc.df[control_col], self.dc.df[treatment_col], metric_type=metric_type, alpha=alpha)
        self.results["ab_test"] = result
        return result

    def test_ab_by_group(self, metric_col, group_col, control_value, treatment_value, metric_type="mean", alpha=0.05):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        df = self.dc.df
        control = df[df[group_col] == control_value][metric_col]
        treatment = df[df[group_col] == treatment_value][metric_col]
        if metric_type == "mean":
            result = ab_test_mean(control, treatment, alpha=alpha)
        elif metric_type == "proportion":
            result = _ab_proportion_test(
                int(control.sum()), int(len(control)),
                int(treatment.sum()), int(len(treatment)),
                alpha
            )
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")
        result["control_group"] = control_value
        result["treatment_group"] = treatment_value
        self.results["ab_test"] = result
        return result

    def test_mutual_info(self, target_col=None):
        if self.dc is None or self.dc.df is None:
            raise ValueError("DataCleaner instance with loaded data required")
        df = self.dc.df
        if target_col is None:
            target_col = self.dc.target_col
        if target_col is None:
            raise ValueError("No target column specified")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        result = mutual_information(X, y)
        self.results["mutual_info"] = result
        return result

    def summary(self):
        lines = []
        if "normality" in self.results:
            lines.append("=== Normality Tests ===")
            for col, res in self.results["normality"].items():
                status = "NORMAL" if res.get("is_normal") else "NOT NORMAL"
                lines.append(f"  {col}: {status} (p={res['p_value']:.4f})")
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
                ci = res.get('ci')
                if ci:
                    lines.append(f"  CI: ({ci[0]:.4f}, {ci[1]:.4f})")
                lines.append(f"  Control: {res.get('control_mean') or res.get('control_rate')} (n={res.get('n_control')})")
                lines.append(f"  Treatment: {res.get('treatment_mean') or res.get('treatment_rate')} (n={res.get('n_treatment')})")
            else:
                lines.append(f"  {res.get('note', 'N/A')}")
        return "\n".join(lines)
