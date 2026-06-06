# DataCleaner

[![PyPI version](https://img.shields.io/pypi/v/clean-data-ml.svg)](https://pypi.org/project/clean-data-ml/)
[![Python versions](https://img.shields.io/pypi/pyversions/clean-data-ml.svg)](https://pypi.org/project/clean-data-ml/)
[![License](https://img.shields.io/pypi/l/clean-data-ml.svg)](https://github.com/MohammadvHossein/clean-data-ml/blob/main/LICENSE)
[![CI](https://github.com/MohammadvHossein/clean-data-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammadvHossein/clean-data-ml/actions/workflows/ci.yml)

**Automated data cleaning & standardization pipeline for ML projects.**

DataCleaner takes raw CSV/Excel data and transforms it into production-ready ML features - handling nulls, encoding categories, selecting the best scaler per column, and packaging everything into a reusable inference pipeline.

---

## Features

| Feature | Description |
|---------|-------------|
| **Input** | CSV & Excel files or in-memory DataFrames |
| **Column dropping** | Pass a list of unwanted columns (IDs, timestamps, etc.) |
| **Target column** | Designate any column as the prediction target |
| **Auto-detect problem type** | Automatically detects classification vs regression from target column |
| **Auto-drop useless columns** | Removes zero-variance, high-cardinality, and duplicated columns |
| **Null handling** | Dynamic threshold: drop rows if nulls are few, KNN-impute if nulls are abundant |
| **Outlier handling** | IQR-based detection with clip or remove options |
| **Encoding** | Auto-detects binary vs multi-category columns; LabelEncoder for binary, OneHotEncoder for categorical |
| **Auto-scaler** | Tests each numeric column for normality & outliers, then picks the optimal scaler (Standard, Robust, MinMax, MaxAbs) |
| **Feature engineering** | Generates polynomial features (interactions, squares) for numeric columns |
| **Imbalance handling** | SMOTE oversampling for imbalanced classification datasets |
| **Train/Val/Test split** | Configurable split ratios |
| **Pipeline export** | Save & reload the full transformation pipeline for inference on new data |
| **Summary** | Quick overview of shape, dtypes, null counts & percentages |
| **Date feature extraction** | Expands datetime columns into year/month/day/dayofweek/weekend |
| **Missing indicators** | Adds `{col}_missing` binary columns for imputed nulls |
| **Feature selection** | Removes weak features via mutual information |
| **Custom encoders/scalers** | Pass your own sklearn encoders and scalers to `prepare()` |
| **Statistical test suite** | Integrated A/B testing, t-tests, z-tests, chi-square, ANOVA |
| **Data profiling** | Self-contained HTML report with distributions, correlations, quality warnings |
| **Schema validation** | Validate column existence and expected dtypes |
| **Duplicate removal** | Drop duplicate rows |

---

## Quick Start

### Install

```bash
pip install clean-data-ml
```

With optional extras:

```bash
pip install clean-data-ml[plot]       # visualization (matplotlib, seaborn)
pip install clean-data-ml[imbalance]   # SMOTE oversampling support
pip install clean-data-ml[all]         # all optional features
```

For a development (editable) install from source:

```bash
git clone https://github.com/MohammadvHossein/clean-data-ml.git
cd clean-data-ml
pip install -e .
pip install -e .[all]                # including all extras
```

### Minimal example

```python
from clean_data_ml import DataCleaner
from sklearn.svm import SVC

dc = DataCleaner()
dc.load("data.csv")
dc.set_target("purchased")
dc.drop_columns(["ID", "timestamp"])

X_train, X_test, y_train, y_test = dc.prepare(test_size=0.2)

model = SVC()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
```

---

## Full Example

### 1. Train & save pipeline

```python
from clean_data_ml import DataCleaner
import pandas as pd
from sklearn.svm import SVC
import joblib

# Sample data
data = pd.DataFrame({
    "ID": range(100),
    "age": [25, 30, 35, None, 40, 45, 50, 55, 60, 65] * 10,
    "salary": [50000, 60000, None, 80000, 90000, 100000, 110000, 120000, None, 140000] * 10,
    "city": ["Tehran", "Shiraz", "Tehran", "Isfahan", None, "Tehran", "Shiraz", "Isfahan", "Tehran", "Shiraz"] * 10,
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"] * 10,
    "purchased": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 10,
})

dc = DataCleaner()
dc.load_df(data)
dc.set_target("purchased")
dc.drop_columns(["ID"])

X_train, X_test, y_train, y_test = dc.prepare(test_size=0.2)

model = SVC(probability=True)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

# Save model & pipeline for later inference
joblib.dump(model, "model.pkl")
dc.save_pipeline("my_pipeline.pkl")
```

### 2. Inference on new data

```python
from clean_data_ml import DataCleaner
import pandas as pd
from sklearn.svm import SVC
import joblib

dc = DataCleaner.load_pipeline("my_pipeline.pkl")
model = joblib.load("model.pkl")

new_data = pd.DataFrame({
    "age": [28, 42, 35],
    "salary": [65000, 95000, 78000],
    "city": ["Tehran", "Isfahan", "Shiraz"],
    "gender": ["F", "M", "F"],
})

processed = dc.transform(new_data)
predictions = model.predict(processed)
probabilities = model.predict_proba(processed)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "Purchased" if pred == 1 else "Not Purchased"
    print(f"Customer {i+1}: {status} (confidence: {max(prob):.2%})")
```

---

## API Reference

### `DataCleaner(random_state=42)`
Main class. All methods return `self` for chaining.

| Method | Description |
|--------|-------------|
| `.load(filepath)` | Load CSV or Excel file |
| `.load_df(df)` | Load from an existing pandas DataFrame |
| `.set_target(col)` | Set the target column |
| `.drop_columns(cols)` | Drop unwanted columns (IDs, etc.) — append-only, safe to call multiple times |
| `.prepare(...)` | Execute the full pipeline (see parameters below) |
| `.get_pipeline()` | Returns the fitted `CleanPipeline` for transforming new data |
| `.save_pipeline(path)` | Save pipeline to disk |
| `.load_pipeline(path)` | Load a saved pipeline — returns a `DataCleaner` instance wrapping the pipeline |
| `.transform(df)` | Apply all cleaning steps to new raw data (same as `get_pipeline().transform(df)`) |
| `.export_cleaned(filepath, include_target=False)` | Export the fully cleaned dataset (features only, or with target if `True`) to CSV or Excel (.xlsx) |
| `.summary()` | Dict with shape, columns, dtypes, null counts |
| `.profile_report(filepath)` | Generate a self-contained HTML data profiling report with stats, distributions, and quality warnings |
| `.drop_duplicates(subset, keep)` | Remove duplicate rows |
| `.validate_schema(expected_schema, required_cols)` | Validate column existence and expected dtypes |
| `.auto_fix_dtypes()` | Auto-convert object columns to datetime or numeric where possible |

### `prepare()` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_size` | `0.2` | Fraction of data for test set |
| `val_size` | `None` | If set, also creates a validation set |
| `handle_nulls` | `True` | Auto-detect and handle missing values |
| `auto_scale` | `True` | Auto-select and apply optimal scaler per column |
| `auto_encode` | `True` | Auto-encode binary (Label) and categorical (OneHot) columns |
| `null_drop_ratio` | `None` | Override dynamic null threshold |
| `auto_drop_useless` | `True` | Drop zero-variance and high-cardinality columns |
| `handle_outliers` | `None` | `"clip"` to cap outliers, `"remove"` to drop, `None` to skip |
| `feature_engineering` | `False` | Add polynomial features (interactions, squares) |
| `handle_imbalance` | `False` | Apply SMOTE oversampling on imbalanced classification data |
| `n_jobs` | `1` | Number of parallel jobs for scaler selection and outlier handling. `-1` uses all cores |
| `extract_date_features` | `False` | Expand datetime columns into year, month, day, dayofweek, weekend |
| `add_missing_indicators` | `False` | Add `{col}_missing` binary columns for imputed nulls |
| `feature_selection` | `None` | `"auto"` (median MI threshold) or a float threshold; removes features below threshold |
| `custom_encoders` | `None` | Dict of `{col: encoder_instance}` to override auto-encoding |
| `custom_scalers` | `None` | Dict of `{col: scaler_instance}` to override auto-scaling |

### `CleanPipeline` (internal)
Holds all fitted transformers. Can be used directly, but prefer `DataCleaner` for full functionality.

| Method | Description |
|--------|-------------|
| `.transform(df)` | Apply all cleaning steps to a raw DataFrame |
| `.save(path)` | Pickle to disk |
| `.load(path)` | Static method -- load from disk |

### Pipeline Attributes (accessible via `dc.pipeline.*`)

| Attribute | Description |
|-----------|-------------|
| `.problem_type` | `"classification"` or `"regression"` (auto-detected) |
| `.dropped_useless_cols` | Columns auto-dropped by `auto_drop_useless` |
| `.outlier_bounds` | IQR bounds used for outlier handling (applied in transform) |
| `.scalers` | Dict of column -> fitted scaler |
| `.onehot_cols` | One-hot encoded column names |
| `.label_encoders` | Binary column -> mapping dict |
| `.feature_cols` | Ordered list of all feature columns after transformation |
| `.poly_features` | Fitted `PolynomialFeatures` transformer (if feature_engineering was enabled) |
| `.custom_encoders` | Dict of user-provided encoders |
| `.custom_scalers` | Dict of user-provided scalers |
| `.cat_impute_values` | Dict of categorical column -> mode used for imputation |
| `.feature_importances_` | Dict of column -> mutual information score (if feature_selection was used) |

---

## How Nulls Are Handled

The threshold for "drop vs impute" is **dynamic** -- it adapts to dataset size:

| Dataset size | Drop threshold | Behavior |
|-------------|---------------|----------|
| 100 rows | 25% | Very conservative -- prefers KNN imputation |
| 1,000 rows | 5% | Balanced approach |
| 10,000+ rows | 1% | More aggressive dropping (plentiful data) |

- **Numeric columns with many nulls** -- `KNNImputer(n_neighbors=5)`
- **Categorical columns with many nulls** -- filled with mode
- **Any column with few nulls** -- those rows are dropped

You can override this with `prepare(null_drop_ratio=0.1)`.

---

## Statistical Test Suite

The `clean_data_ml.stats` module provides a comprehensive set of statistical tests for data analysis:

### Standalone Functions

| Function | Description |
|----------|-------------|
| `normality_test(series, method)` | Shapiro-Wilk, D'Agostino, or Anderson-Darling normality test |
| `correlation_test(x, y, method)` | Pearson, Spearman, or Kendall correlation |
| `ks_test(a, b)` | Kolmogorov-Smirnov (two-sample distribution test) |
| `chi_square_test(a, b)` | Chi-square test of independence |
| `variance_test(a, b, method)` | Levene, Bartlett, or Fligner test for equal variance |
| `anova_one_way(*groups)` | One-way ANOVA |
| `z_test_one_sample(series, pop_mean)` | One-sample z-test for mean |
| `z_test_two_sample(a, b)` | Two-sample z-test for mean |
| `z_test_proportion(successes, n, p)` | One-sample proportion z-test |
| `z_test_two_proportion(s1, n1, s2, n2)` | Two-sample proportion z-test |
| `t_test_one_sample(series, pop_mean)` | One-sample t-test |
| `t_test_independent(a, b)` | Independent two-sample t-test |
| `t_test_paired(a, b)` | Paired t-test |
| `ab_test_mean(control, treatment)` | A/B test on means (lift, CI, significance) |
| `ab_test_proportion(control, treatment)` | A/B test on proportions |
| `mutual_information(X, y)` | Mutual Information between features and target |

### StatisticalTestSuite (integration with DataCleaner)

```python
from clean_data_ml import DataCleaner, stats

dc = DataCleaner()
dc.load_df(data).set_target("purchased")

suite = stats.StatisticalTestSuite(dc)
suite.test_normality()
suite.test_correlations(target_col="purchased")
suite.test_chi_square("gender", "city")
suite.test_anova("age", "city")
suite.test_z_one_sample("age", pop_mean=35)
suite.test_t_independent("age", "score")
suite.test_ab_by_group("converted", "group", "A", "B", metric_type="proportion")
print(suite.summary())
```

## Visualization Module

The `clean_data_ml.plotting` module (requires `pip install -e .[plot]`):

| Function | Description |
|----------|-------------|
| `plot_null_report(dc)` | Bar charts of null counts and percentages |
| `plot_distributions(dc, cols)` | Histograms + boxplots for numeric columns |
| `plot_correlation(dc)` | Correlation heatmap |
| `plot_before_after(dc)` | Compare raw vs cleaned distributions |

## Project Structure

```
clean_data_ml/
  __init__.py       Package exports
  cleaner.py        DataCleaner + CleanPipeline classes
  auto_scaler.py    Automatic scaler selection logic
  stats.py          Statistical test suite (t-test, z-test, AB test, etc.)
  plotting.py       Optional visualization module
setup.py                  Package metadata
pyproject.toml            Build configuration
MANIFEST.in               sdist inclusion rules
LICENSE                   MIT license
example_train.py          Training example
example_inference.py      Inference example
.pre-commit-config.yaml   Linting hooks (black, isort, flake8)
.gitignore                Ignored files
README.md                 This file
tests/
    conftest.py           Shared test fixtures
    test_cleaner.py       DataCleaner / CleanPipeline tests
    test_auto_scaler.py   Scaler selection tests
    test_stats.py         Statistical test suite tests
    test_plotting.py      Visualization module tests
```

## Additional Features

### Auto-Drop Useless Columns

The library automatically detects and removes:
- **Zero-variance columns** -- columns with a single unique value
- **High-cardinality columns** -- non-numeric columns where unique values exceed 90% of rows (e.g., free-text fields)

Disabled with `prepare(auto_drop_useless=False)`.

### Outlier Handling (IQR)

After null handling, each numeric column is checked using the Interquartile Range method:
- **Lower bound**: Q1 - 1.5 x IQR
- **Upper bound**: Q3 + 1.5 x IQR

Two modes:
- `"clip"` -- caps values at the bounds (preserves row count)
- `"remove"` -- drops rows with outliers

Activated with `prepare(handle_outliers="clip")`.

### Feature Engineering

Generates polynomial features (degree 2) for numeric columns with more than 2 unique values. Creates interaction terms and squared features automatically.

Activated with `prepare(feature_engineering=True)`.

### Date Feature Extraction

When `prepare(extract_date_features=True)`, datetime columns are automatically expanded into numerical components:
- `{col}_year`, `{col}_month`, `{col}_day`, `{col}_dayofweek`, `{col}_weekend`
- The original datetime column is dropped afterward.

This happens early in the pipeline so the derived numeric columns benefit from all subsequent steps (encoding, scaling, feature engineering, etc.).

### Missing Indicators

When `prepare(add_missing_indicators=True)`, for every column that receives KNN imputation (null ratio above threshold), an additional binary column `{col}_missing` is added, flagging which rows originally contained nulls. This lets the model learn patterns from the missingness itself.

### Feature Selection

Controlled by `prepare(feature_selection="auto")` or `prepare(feature_selection=0.01)`.

After all transformations, Mutual Information is computed between each feature and the target. Features with MI below the threshold are dropped:
- `"auto"` -- drops features below the **median** MI score
- `float` (e.g., `0.01`) -- drops features below that absolute threshold

Set to `None` (default) to skip feature selection entirely.

### Custom Encoders & Custom Scalers

Pass fitted or unfitted sklearn-compatible transformers to override auto-detection:

```python
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer

dc.prepare(
    custom_encoders={"city": OrdinalEncoder()},
    custom_scalers={"salary": KBinsDiscretizer(n_bins=5, encode="ordinal")},
)
```

These are stored in `dc.pipeline.custom_encoders` / `dc.pipeline.custom_scalers` and applied during `transform()` as well.

### Imbalanced Data (SMOTE)

When `handle_imbalance=True` and the problem is classification, SMOTE oversampling is applied to the training set after the train/test split. Requires `imbalanced-learn`:

```bash
pip install imbalanced-learn
```

---

## How Scaler Selection Works

For each numeric column, the library tests:

1. **Normality** -- Shapiro-Wilk test (p > 0.05 => normal)
2. **Outliers** -- IQR method (1.5x IQR rule)
3. **Bounds** -- min >= 0 & max <= 1
4. **Sparsity** -- >40% zeros

**Note:** Tree-based models (Random Forest, XGBoost, LightGBM, etc.) do not require scaling or normalization -- they split on thresholds and are invariant to monotonic transformations. Scaling is only needed for distance-based or gradient-based models (SVM, KNN, Neural Networks, Logistic Regression, etc.). You can skip auto-scaling with `prepare(auto_scale=False)` if using a tree-based model.

Then assigns the optimal scaler:

| Condition | Scaler |
|-----------|--------|
| Normal + no outliers | `StandardScaler` |
| Has outliers | `RobustScaler` |
| Bounded [0, 1] | `MinMaxScaler` |
| Sparse | `MaxAbsScaler` |
| Default | `StandardScaler` |

---

## Requirements

- Python >= 3.8
- pandas >= 1.3
- numpy >= 1.21
- scikit-learn >= 1.0
- scipy >= 1.7
- joblib
- openpyxl (for Excel support)

---

## License

MIT
