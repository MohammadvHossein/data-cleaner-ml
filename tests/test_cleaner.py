"""Tests for cleaner module — DataCleaner and CleanPipeline."""

import numpy as np
import pandas as pd
import pytest

from clean_data_ml import DataCleaner


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
    dc.df["raw_date"] = pd.Series(["2024-01-01"] * len(dc.df), dtype=object)
    dc.df["raw_num"] = pd.Series(["1,234"] * len(dc.df), dtype=object)
    fixes = dc.auto_fix_dtypes()
    assert len([f for f in fixes if "numeric" in f]) >= 1


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


# ---------------------------------------------------------------------------
# Layer 2 — Date feature extraction
# ---------------------------------------------------------------------------

def test_extract_date_features():
    np.random.seed(42)
    n = 50
    start = "2023-01-01"
    df = pd.DataFrame({
        "join_date": pd.date_range(start, periods=n, freq="D"),
        "value": np.random.rand(n),
        "target": np.random.choice([0, 1], n),
    })
    dc = DataCleaner(random_state=42)
    dc.load_df(df).set_target("target")
    dc.prepare(test_size=0.3, extract_date_features=True)
    p = dc.pipeline
    assert "join_date" in p.date_cols
    # year is dropped by auto_drop_useless if constant
    for suffix in ("month", "day", "dayofweek", "weekend"):
        assert f"join_date_{suffix}" in dc.X_clean.columns
    assert "join_date" not in dc.X_clean.columns


def test_extract_date_features_in_transform():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "join_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "value": np.random.rand(n),
        "target": np.random.choice([0, 1], n),
    })
    dc = DataCleaner(random_state=42)
    dc.load_df(df).set_target("target")
    dc.prepare(test_size=0.3, extract_date_features=True)
    new = dc.transform(pd.DataFrame({
        "join_date": pd.to_datetime(["2024-06-15"]),
        "value": [0.5],
    }))
    # Non-constant date features should appear
    assert "join_date_month" in new.columns
    assert new["join_date_month"].iloc[0] == 6


# ---------------------------------------------------------------------------
# Layer 2 — Missing indicators
# ---------------------------------------------------------------------------

def test_missing_indicators(dc):
    dc.prepare(test_size=0.2, null_drop_ratio=0.01, add_missing_indicators=True)
    p = dc.pipeline
    assert len(p.missing_indicator_cols) > 0
    for col in p.missing_indicator_cols:
        assert f"{col}_missing" in dc.X_clean.columns


# ---------------------------------------------------------------------------
# Layer 2 — Feature selection
# ---------------------------------------------------------------------------

def test_feature_selection_auto(dc):
    dc.drop_columns(["gender"])
    dc.prepare(test_size=0.2, feature_selection="auto")
    p = dc.pipeline
    assert p.feature_selection_threshold is not None
    assert isinstance(p.feature_selection_removed, list)


def test_feature_selection_no_drop_when_all_high():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "x1": np.random.rand(n),
        "x2": np.random.rand(n),
        "target": np.random.choice([0, 1], n),
    })
    dc = DataCleaner(random_state=42)
    dc.load_df(df).set_target("target")
    dc.prepare(test_size=0.3, feature_selection=0.0)
    assert len(dc.pipeline.feature_selection_removed) == 0


# ---------------------------------------------------------------------------
# Layer 2 — Custom encoder / scaler
# ---------------------------------------------------------------------------

def test_custom_encoder():
    from sklearn.preprocessing import LabelEncoder
    np.random.seed(42)
    df = pd.DataFrame({
        "color": np.random.choice(["red", "green", "blue"], 100),
        "value": np.random.rand(100),
        "target": np.random.choice([0, 1], 100),
    })
    dc = DataCleaner(random_state=42)
    dc.load_df(df).set_target("target")
    dc.prepare(test_size=0.3, custom_encoders={"color": LabelEncoder()})
    assert dc.X_clean["color"].dtype in (np.int32, np.int64)


def test_custom_scaler():
    from sklearn.preprocessing import StandardScaler
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.normal(35, 10, 100),
        "target": np.random.choice([0, 1], 100),
    })
    dc = DataCleaner(random_state=42)
    dc.load_df(df).set_target("target")
    dc.prepare(test_size=0.3, custom_scalers={"age": StandardScaler()})
    assert abs(dc.X_clean["age"].mean()) < 1e-10
    assert abs(dc.X_clean["age"].std() - 1.0) < 0.2
    assert "age" in dc.pipeline.custom_scalers


# ---------------------------------------------------------------------------
# Layer 3 — __main__.py
# ---------------------------------------------------------------------------

def test_main_module():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "clean_data_ml"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "clean_data_ml v" in result.stdout
