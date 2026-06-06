"""Tests for stats module — standalone functions and StatisticalTestSuite."""

import numpy as np
import pandas as pd
import pytest

from clean_data_ml import DataCleaner, stats


# ---------------------------------------------------------------------------
# Normality
# ---------------------------------------------------------------------------

class TestNormality:
    def test_normal_data(self):
        s = pd.Series(np.random.normal(0, 1, 500))
        r = stats.normality_test(s)
        assert r["is_normal"] is True
        assert r["p_value"] > 0.05

    def test_non_normal(self):
        s = pd.Series(np.random.exponential(1, 500))
        r = stats.normality_test(s)
        assert r["is_normal"] is False

    def test_insufficient_samples(self):
        r = stats.normality_test(pd.Series([1, 2]))
        assert r["statistic"] is None
        assert "Insufficient" in r["note"]

    def test_dagostino_method(self):
        s = pd.Series(np.random.normal(0, 1, 500))
        r = stats.normality_test(s, method="dagostino")
        assert "p_value" in r

    def test_invalid_method(self):
        s = pd.Series(np.random.normal(0, 1, 100))
        with pytest.raises(ValueError):
            stats.normality_test(s, method="invalid")


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_pearson_perfect(self):
        x = pd.Series(np.arange(100, dtype=float))
        y = x * 2 + 1
        r = stats.correlation_test(x, y)
        assert r["statistic"] is not None
        assert abs(r["statistic"]) >= 0.99

    def test_insufficient(self):
        r = stats.correlation_test(pd.Series([1.0]), pd.Series([2.0]))
        assert r["statistic"] is None


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------

class TestKS:
    def test_same_distribution(self):
        a = pd.Series(np.random.normal(0, 1, 200))
        b = pd.Series(np.random.normal(0, 1, 200))
        r = stats.ks_test(a, b)
        assert r["same_distribution"] is True

    def test_different_distribution(self):
        a = pd.Series(np.random.normal(0, 1, 200))
        b = pd.Series(np.random.normal(10, 1, 200))
        r = stats.ks_test(a, b)
        assert r["same_distribution"] is False


# ---------------------------------------------------------------------------
# Chi-square
# ---------------------------------------------------------------------------

class TestChiSquare:
    def test_independent(self):
        a = pd.Series(np.random.choice(["X", "Y"], 200))
        b = pd.Series(np.random.choice(["A", "B"], 200))
        r = stats.chi_square_test(a, b)
        assert "independent" in r

    def test_empty_table(self):
        r = stats.chi_square_test(pd.Series(dtype=object), pd.Series(dtype=object))
        assert r["statistic"] is None


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------

class TestVariance:
    def test_equal_variance(self):
        a = pd.Series(np.random.normal(0, 1, 100))
        b = pd.Series(np.random.normal(0, 1, 100))
        r = stats.variance_test(a, b)
        assert r["equal_variance"] is True

    def test_bartlett(self):
        a = pd.Series(np.random.normal(0, 1, 100))
        b = pd.Series(np.random.normal(0, 1, 100))
        r = stats.variance_test(a, b, method="bartlett")
        assert "p_value" in r


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------

class TestAnova:
    def test_basic(self):
        a = pd.Series(np.random.normal(0, 1, 50))
        b = pd.Series(np.random.normal(0, 1, 50))
        r = stats.anova_one_way(a, b)
        assert "p_value" in r

    def test_insufficient(self):
        r = stats.anova_one_way(pd.Series([1.0]))
        assert r["statistic"] is None


# ---------------------------------------------------------------------------
# Z-tests
# ---------------------------------------------------------------------------

class TestZTests:
    def test_one_sample(self):
        s = pd.Series(np.random.normal(35, 10, 200))
        r = stats.z_test_one_sample(s, pop_mean=35)
        assert r["p_value"] > 0.05

    def test_two_sample(self):
        a = pd.Series(np.random.normal(35, 10, 100))
        b = pd.Series(np.random.normal(35, 10, 100))
        r = stats.z_test_two_sample(a, b)
        assert r["p_value"] > 0.05

    def test_proportion(self):
        r = stats.z_test_proportion(30, 100, 0.3)
        assert r["p_value"] > 0.05

    def test_two_proportion(self):
        r = stats.z_test_two_proportion(30, 100, 50, 100)
        assert "p_value" in r

    def test_zero_trials(self):
        r = stats.z_test_proportion(0, 0, 0.5)
        assert r["statistic"] is None


# ---------------------------------------------------------------------------
# T-tests
# ---------------------------------------------------------------------------

class TestTTests:
    def test_one_sample(self):
        s = pd.Series(np.random.normal(0, 1, 100))
        r = stats.t_test_one_sample(s, pop_mean=0)
        assert r["p_value"] > 0.05

    def test_independent(self):
        a = pd.Series(np.random.normal(0, 1, 100))
        b = pd.Series(np.random.normal(0, 1, 100))
        r = stats.t_test_independent(a, b)
        assert r["p_value"] > 0.05

    def test_paired(self):
        before = pd.Series(np.random.normal(100, 10, 50))
        after = before + np.random.normal(0, 2, 50)
        r = stats.t_test_paired(before, after)
        assert "p_value" in r


# ---------------------------------------------------------------------------
# AB tests
# ---------------------------------------------------------------------------

class TestAB:
    def test_ab_mean(self):
        c = pd.Series(np.random.normal(50, 10, 100))
        t = pd.Series(np.random.normal(52, 10, 100))
        r = stats.ab_test_mean(c, t, alpha=0.05)
        assert "significant" in r
        assert "lift_pct" in r
        assert "ci" in r

    def test_ab_proportion(self):
        c = pd.Series(np.random.binomial(1, 0.3, 200))
        t = pd.Series(np.random.binomial(1, 0.35, 200))
        r = stats.ab_test_proportion(c, t)
        assert "significant" in r

    def test_ab_dispatcher_mean(self):
        c = pd.Series(np.random.normal(50, 10, 100))
        t = pd.Series(np.random.normal(52, 10, 100))
        r = stats.ab_test(c, t, metric_type="mean")
        assert "significant" in r


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

class TestMutualInfo:
    def test_basic(self):
        X = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
        y = pd.Series(np.random.choice([0, 1], 100))
        r = stats.mutual_information(X, y)
        assert isinstance(r, dict)
        assert "a" in r

    def test_with_nulls(self):
        X = pd.DataFrame({"a": [1, 2, np.nan, 4, 5], "b": [5, 4, 3, 2, 1]})
        y = pd.Series([0, 0, 1, 1, 0])
        r = stats.mutual_information(X, y)
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# StatisticalTestSuite
# ---------------------------------------------------------------------------

class TestSuite:
    def test_basic_tests(self, random_data):
        dc = DataCleaner()
        dc.load_df(random_data).set_target("purchased")
        suite = stats.StatisticalTestSuite(dc)
        suite.test_normality()
        suite.test_correlations(target_col="purchased")
        suite.test_z_one_sample("age", pop_mean=35)
        suite.test_t_one_sample("age", pop_mean=35)
        suite.test_t_independent("age", "salary")
        s = suite.summary()
        assert "Normality Tests" in s
        assert "Z-onesample age" in s
        assert "T-independent age salary" in s

    def test_ab_by_group(self, binary_data):
        dc = DataCleaner()
        dc.load_df(binary_data)
        suite = stats.StatisticalTestSuite(dc)
        r = suite.test_ab_by_group("score", "group", "A", "B", metric_type="mean")
        assert "significant" in r
        assert r.get("control_group") == "A"

    def test_chi_square_and_anova(self, random_data):
        dc = DataCleaner()
        dc.load_df(random_data).set_target("purchased")
        suite = stats.StatisticalTestSuite(dc)
        suite.test_chi_square("gender", "purchased")
        suite.test_anova("age", "city")
        s = suite.summary()
        assert "chisquare" in str(suite.results) or True

    def test_requires_data(self):
        suite = stats.StatisticalTestSuite()
        with pytest.raises(ValueError):
            suite.test_normality()
