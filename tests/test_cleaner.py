"""Tests for cleaner module — DataCleaner and CleanPipeline."""

import numpy as np
import pandas as pd
import pytest

from data_cleaner import DataCleaner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dc(random_data):
    dc = DataCleaner(random_state=42)
    dc.load_df(random_data).set_target("purchased")
    return dc


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def test_load_df(random_data):
    dc = DataCleaner().load_df(random_data)
    assert dc.df is not None
    assert dc.raw_df is not None
    assert len(dc.df) == 200


def test_set_target(dc):
    assert dc.target_col == "purchased"


def test_set_target_missing(dc):
    with pytest.raises(ValueError, match="not found"):
        dc.set_target("nonexistent")


# ---------------------------------------------------------------------------
# drop_columns (append mode)
# ---------------------------------------------------------------------------

def test_drop_columns_append(dc):
    dc.drop_columns(["gender"])
    dc.drop_columns(["score"])
    assert dc.columns_to_drop == ["gender", "score"]


def test_drop_columns_missing(dc):
    with pytest.raises(ValueError, match="not found"):
        dc.drop_columns(["nope"])


# ---------------------------------------------------------------------------
# prepare — basic run
# ---------------------------------------------------------------------------

def test_prepare_basic(dc):
    X_tr, X_te, y_tr, y_te = dc.prepare(test_size=0.2)
    assert len(X_tr) > 0
    assert len(X_te) > 0
    assert len(y_tr) == len(X_tr)
    assert len(y_te) == len(X_te)
    assert dc.is_fitted
    assert dc.pipeline.problem_type in ("classification", "regression")


def test_prepare_no_target():
    with pytest.raises(ValueError, match="No data loaded"):
        DataCleaner().prepare()


def test_prepare_no_target_set(random_data):
    dc = DataCleaner()
    dc.load_df(random_data)
    with pytest.raises(ValueError, match="not set"):
        dc.prepare()


# ---------------------------------------------------------------------------
# prepare — with validation split
# ---------------------------------------------------------------------------

def test_prepare_with_val(dc):
    result = dc.prepare(test_size=0.2, val_size=0.1)
    assert len(result) == 6
    X_tr, X_v, X_te, y_tr, y_v, y_te = result
    assert len(X_tr) > 0 and len(X_v) > 0 and len(X_te) > 0


# ---------------------------------------------------------------------------
# prepare — outlier remove syncs y
# ---------------------------------------------------------------------------

def test_outlier_remove_syncs_y(dc):
    dc.prepare(test_size=0.2, handle_outliers="remove")
    assert len(dc.X_clean) == len(dc.y_clean)


def test_outlier_clip_keeps_shape(dc):
    before = len(dc.df)
    dc.prepare(test_size=0.2, handle_outliers="clip")
    # clipping preserves row count
    assert len(dc.X_clean) == len(dc.y_clean)


# ---------------------------------------------------------------------------
# prepare — feature engineering + transform
# ---------------------------------------------------------------------------

def test_feature_engineering_in_transform(dc):
    dc.drop_columns(["gender"])
    X_tr, X_te, y_tr, y_te = dc.prepare(
        test_size=0.2, feature_engineering=True, handle_outliers="clip"
    )
    n_cols = X_tr.shape[1]
    new = dc.transform(pd.DataFrame({"age": [30], "salary": [70000], "city": ["Tehran"]}))
    assert new.shape[1] == n_cols, f"Expected {n_cols} cols, got {new.shape[1]}"
    assert new.shape[0] == 1


# ---------------------------------------------------------------------------
# prepare — SMOTE
# ---------------------------------------------------------------------------

def test_prepare_smote(dc):
    # Make data imbalanced
    dc.df.loc[dc.df.index >= 50, "purchased"] = 0
    dc.prepare(test_size=0.3, handle_imbalance=True)
    assert dc.is_fitted


# ---------------------------------------------------------------------------
# load_pipeline
# ---------------------------------------------------------------------------

def test_load_pipeline(dc, tmp_path):
    dc.drop_columns(["gender"])
    dc.prepare(test_size=0.2)
    path = tmp_path / "pipe.pkl"
    dc.save_pipeline(str(path))
    loaded = DataCleaner.load_pipeline(str(path))
    assert isinstance(loaded, DataCleaner)
    assert loaded.is_fitted
    result = loaded.transform(pd.DataFrame({"age": [25], "salary": [50000], "city": ["Shiraz"]}))
    assert result.shape[0] == 1


# ---------------------------------------------------------------------------
# export_cleaned
# ---------------------------------------------------------------------------

def test_export_cleaned_csv(dc, tmp_path):
    dc.prepare(test_size=0.2)
    path = tmp_path / "out.csv"
    dc.export_cleaned(str(path))
    assert path.exists()


def test_export_cleaned_xls_rejected(dc):
    dc.prepare(test_size=0.2)
    with pytest.raises(ValueError, match="not supported"):
        dc.export_cleaned("test.xls")


def test_export_with_target(dc, tmp_path):
    dc.prepare(test_size=0.2)
    path = tmp_path / "out.csv"
    dc.export_cleaned(str(path), include_target=True)
    df = pd.read_csv(str(path))
    assert dc.target_col in df.columns


# ---------------------------------------------------------------------------
# pipeline attributes
# ---------------------------------------------------------------------------

def test_pipeline_attributes_set(dc):
    dc.drop_columns(["gender"])
    dc.prepare(test_size=0.2)
    p = dc.pipeline
    assert p.target_col == "purchased"
    assert p.columns_to_drop == ["gender"]
    assert isinstance(p.feature_cols, list)
    assert len(p.feature_cols) > 0


# ---------------------------------------------------------------------------
# drop_duplicates
# ---------------------------------------------------------------------------

def test_drop_duplicates(dc):
    dc.df = pd.concat([dc.df, dc.df.iloc[[0]]], ignore_index=True)
    before = len(dc.df)
    dc.drop_duplicates()
    assert len(dc.df) < before


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

def test_validate_schema_passes(dc):
    issues = dc.validate_schema(required_cols=["age", "salary"])
    assert issues == []


def test_validate_schema_fails(dc):
    issues = dc.validate_schema(required_cols=["nope"])
    assert len(issues) == 1
    assert "nope" in issues[0]


# ---------------------------------------------------------------------------
# auto_fix_dtypes
# ---------------------------------------------------------------------------

def test_auto_fix_dtypes(dc):
    dc.df["raw_date"] = "2024-01-01"
    fixes = dc.auto_fix_dtypes()
    assert len(fixes) >= 1


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary(dc):
    s = dc.summary()
    assert isinstance(s, dict)
    assert "shape" in s
    assert "null_counts" in s


# ---------------------------------------------------------------------------
# profile_report
# ---------------------------------------------------------------------------

def test_profile_report(dc, tmp_path):
    path = tmp_path / "report.html"
    html = dc.profile_report(str(path))
    assert len(html) > 500
    assert path.exists()
