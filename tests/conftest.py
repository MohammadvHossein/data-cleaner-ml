"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest

SEED = 42


@pytest.fixture
def random_data():
    """Return a simple DataFrame with mixed types and nulls."""
    np.random.seed(SEED)
    n = 200
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, n).astype(int),
        "salary": np.random.exponential(80000, n),
        "score": np.random.uniform(0, 100, n),
        "city": np.random.choice(["Tehran", "Shiraz", "Isfahan", "Mashhad", "Tabriz"], n),
        "gender": np.random.choice(["M", "F"], n),
        "purchased": np.random.choice([0, 1], n),
    })
    df.loc[::10, "age"] = None
    df.loc[::15, "city"] = None
    return df


@pytest.fixture
def binary_data():
    """Return a DataFrame with 0/1 columns for proportion tests."""
    np.random.seed(SEED)
    return pd.DataFrame({
        "converted": np.random.binomial(1, 0.3, 200),
        "group": np.random.choice(["A", "B"], 200),
        "score": np.random.uniform(0, 100, 200),
    })
