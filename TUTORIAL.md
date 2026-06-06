# DataCleaner Complete Tutorial

**DataCleaner** is an automated data cleaning, standardization, and ML pipeline preparation library for Python.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Loading Data](#loading-data)
4. [Configuration](#configuration)
5. [The prepare() Method and All Parameters](#the-prepare-method-and-all-parameters)
6. [Null Value Handling](#null-value-handling)
7. [Outlier Handling](#outlier-handling)
8. [Auto-Drop Useless Columns](#auto-drop-useless-columns)
9. [Automatic Scaler Selection](#automatic-scaler-selection)
10. [Automatic Encoding](#automatic-encoding)
11. [Feature Engineering](#feature-engineering)
12. [Date Feature Extraction](#date-feature-extraction)
13. [Missing Indicators](#missing-indicators)
14. [Feature Selection](#feature-selection)
15. [Custom Encoders and Scalers](#custom-encoders-and-scalers)
16. [Imbalanced Data Handling (SMOTE)](#imbalanced-data-handling-smote)
17. [Saving and Loading the Pipeline](#saving-and-loading-the-pipeline)
18. [Inference on New Data](#inference-on-new-data)
19. [Data Profiling Report](#data-profiling-report)
20. [Schema Validation](#schema-validation)
21. [Auto-Fix Data Types](#auto-fix-data-types)
22. [Removing Duplicate Rows](#removing-duplicate-rows)
23. [Exporting Cleaned Data](#exporting-cleaned-data)
24. [Statistical Tests (Stats Module)](#statistical-tests-stats-module)
25. [Visualization Module (Plotting)](#visualization-module-plotting)
26. [Complete End-to-End Example](#complete-end-to-end-example)
27. [Tips and Best Practices](#tips-and-best-practices)

---

## Installation

### Install from PyPI

```bash
pip install data-cleaner
```

### Install with Optional Extras

```bash
pip install data-cleaner[plot]       # visualization (matplotlib, seaborn)
pip install data-cleaner[imbalance]   # SMOTE oversampling support
pip install data-cleaner[all]         # all optional features
```

### Install from Source

```bash
git clone https://github.com/MohammadvHossein/data_cleaner.git
cd data_cleaner
pip install -e .
pip install -e .[all]
```

---

## Quick Start

```python
from data_cleaner import DataCleaner
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

## Loading Data

### From CSV or Excel File

```python
dc = DataCleaner()
dc.load("data.csv")              # CSV file
dc.load("data.xlsx")             # Excel file
```

### From an In-Memory DataFrame

```python
import pandas as pd
df = pd.DataFrame({
    "age": [25, 30, None, 40],
    "salary": [50000, 60000, 80000, 90000],
    "city": ["Tehran", "Shiraz", "Isfahan", "Tehran"],
    "purchased": [1, 0, 1, 0],
})
dc.load_df(df)
```

---

## Configuration

### Setting the Target Column

```python
dc.set_target("purchased")
```

### Dropping Unwanted Columns

`drop_columns()` is safe to call multiple times -- columns are appended, never replaced.

```python
dc.drop_columns(["ID", "timestamp"])
dc.drop_columns(["unnecessary_feature"])   # appended, not replaced
```

---

## The prepare() Method and All Parameters

The `prepare()` method is the core of the library. It runs all cleaning steps in a fixed order:

1. Drop configured columns
2. Auto-detect problem type (classification / regression)
3. Extract date features (if enabled)
4. Auto-drop useless columns
5. Detect column types (numeric, categorical, binary)
6. Handle nulls
7. Add missing indicators (if enabled)
8. Handle outliers
9. Feature engineering (if enabled)
10. Encode columns
11. Scale columns
12. Feature selection (if enabled)
13. Train/Test split
14. SMOTE (if enabled)

```python
X_train, X_test, y_train, y_test = dc.prepare(
    test_size=0.2,                # fraction of data for testing
    val_size=None,                # if set, also creates a validation set
    handle_nulls=True,            # auto-handle missing values
    auto_scale=True,              # auto-select optimal scaler per column
    auto_encode=True,             # auto-encode binary and categorical columns
    null_drop_ratio=None,         # override the dynamic null threshold
    auto_drop_useless=True,       # drop zero-variance / high-cardinality cols
    handle_outliers=None,         # "clip" or "remove" for outlier handling
    feature_engineering=False,    # add polynomial features
    handle_imbalance=False,       # SMOTE for imbalanced classification
    n_jobs=1,                     # parallel workers for scaler selection and outliers
    extract_date_features=False,  # expand datetime cols into numeric components
    add_missing_indicators=False, # add {col}_missing binary columns
    feature_selection=None,       # "auto" or a float threshold
    custom_encoders=None,         # custom sklearn encoders {col: encoder}
    custom_scalers=None,          # custom sklearn scalers {col: scaler}
)
```

### Return Values

```python
# Default (train/test split)
X_train, X_test, y_train, y_test = dc.prepare(test_size=0.2)

# With validation set
X_train, X_val, X_test, y_train, y_val, y_test = dc.prepare(val_size=0.15)
```

---

## Null Value Handling

### Dynamic Threshold

The threshold for "drop vs impute" adapts to dataset size:

| Dataset Size | Drop Threshold | Behavior |
|-------------|---------------|----------|
| 100 rows | 25% | Very conservative -- prefers KNN imputation |
| 1,000 rows | 5% | Balanced approach |
| 10,000+ rows | 1% | More aggressive dropping |

You can override the threshold:

```python
X_train, X_test, y_train, y_test = dc.prepare(null_drop_ratio=0.1)
```

### How It Works

- **Numeric columns with many nulls**: `KNNImputer` with 5 neighbors
- **Categorical columns with many nulls**: filled with mode (most frequent value)
- **Columns with few nulls**: those rows are dropped

### Nulls in the Target Column

If the target column has fewer than 5% nulls, those rows are dropped. If more than 5%, an error is raised:

```python
# This will raise an error if the target has >5% nulls
# ValueError: Target column 'price' has 12.5% null values. Please clean it manually.
```

---

## Outlier Handling

Uses the IQR (Interquartile Range) method:

- **Lower bound**: Q1 - 1.5 x IQR
- **Upper bound**: Q3 + 1.5 x IQR

Two modes:

```python
# Mode 1: clip -- caps outliers at bounds, preserves row count
X_train, X_test, y_train, y_test = dc.prepare(handle_outliers="clip")

# Mode 2: remove -- drops rows with outliers
X_train, X_test, y_train, y_test = dc.prepare(handle_outliers="remove")

# Disabled
X_train, X_test, y_train, y_test = dc.prepare(handle_outliers=None)
```

Outliers are handled after null imputation and before feature engineering.

---

## Auto-Drop Useless Columns

The library automatically detects and removes:

- **Zero-variance columns**: columns with a single unique value
- **High-cardinality columns**: non-numeric columns where unique values exceed 90% of rows (e.g., free-text fields)

```python
# Disable
X_train, X_test, y_train, y_test = dc.prepare(auto_drop_useless=False)
```

Dropped columns are stored in `dc.pipeline.dropped_useless_cols`.

---

## Automatic Scaler Selection

For each numeric column, the library runs these tests:

1. **Normality** -- Shapiro-Wilk test (p > 0.05 means normal)
2. **Outliers** -- IQR method (1.5x IQR rule)
3. **Bounds** -- min >= 0 and max <= 1
4. **Sparsity** -- more than 40% zeros

Then selects the best scaler:

| Condition | Scaler |
|-----------|--------|
| Normal + no outliers | `StandardScaler` |
| Has outliers | `RobustScaler` |
| Bounded [0, 1] | `MinMaxScaler` |
| Sparse | `MaxAbsScaler` |
| Default | `StandardScaler` |

**Note**: Tree-based models (Random Forest, XGBoost, LightGBM, etc.) do not need scaling. Disable it with `auto_scale=False`.

```python
X_train, X_test, y_train, y_test = dc.prepare(auto_scale=False)
```

View the selected scalers:

```python
for col, scaler in dc.pipeline.scalers.items():
    print(f"{col}: {type(scaler).__name__}")
```

---

## Automatic Encoding

The library auto-detects:

- **Binary columns** (2 unique values) -> `LabelEncoder`
- **Categorical columns** (3+ unique values) -> `OneHotEncoder`

```python
# Disable auto-encoding
X_train, X_test, y_train, y_test = dc.prepare(auto_encode=False)
```

If auto-encoding is disabled, all non-numeric columns are encoded with `LabelEncoder`.

---

## Feature Engineering

Enable `feature_engineering=True` to generate polynomial features (degree 2) for numeric columns with more than 2 unique values:

```python
X_train, X_test, y_train, y_test = dc.prepare(feature_engineering=True)
```

This creates interaction terms (column A x column B) and squared terms (column^2). These features are also generated during `transform()` for inference.

---

## Date Feature Extraction

When `extract_date_features=True`, datetime columns are expanded into numeric components:

- `{col}_year`
- `{col}_month`
- `{col}_day`
- `{col}_dayofweek` (0=Monday)
- `{col}_weekend` (0 or 1)

The original datetime column is dropped afterward.

```python
X_train, X_test, y_train, y_test = dc.prepare(extract_date_features=True)
```

This step runs **before** auto-drop-useless (so derived columns survive) and **before** type detection (so the new numeric columns are classified correctly).

---

## Missing Indicators

When `add_missing_indicators=True`, for every column that receives KNN imputation, a binary column `{col}_missing` is added, flagging which rows originally contained nulls:

```python
X_train, X_test, y_train, y_test = dc.prepare(add_missing_indicators=True)
```

This lets the model learn patterns from missingness itself.

---

## Feature Selection

After all transformations, Mutual Information (MI) is computed between each feature and the target. Weak features are dropped:

```python
# Auto mode -- drops features below the median MI score
X_train, X_test, y_train, y_test = dc.prepare(feature_selection="auto")

# With a numeric threshold -- drops features with MI below 0.01
X_train, X_test, y_train, y_test = dc.prepare(feature_selection=0.01)

# Disabled
X_train, X_test, y_train, y_test = dc.prepare(feature_selection=None)
```

Diagnostic info:

```python
print("Selection threshold:", dc.pipeline.feature_selection_threshold)
print("Dropped columns:", dc.pipeline.feature_selection_removed)
```

---

## Custom Encoders and Scalers

You can pass your own sklearn-compatible transformers to `prepare()`:

### Custom Encoders

```python
from sklearn.preprocessing import OrdinalEncoder

X_train, X_test, y_train, y_test = dc.prepare(
    custom_encoders={"city": OrdinalEncoder()},
)
```

### Custom Scalers

```python
from sklearn.preprocessing import KBinsDiscretizer

X_train, X_test, y_train, y_test = dc.prepare(
    custom_scalers={"salary": KBinsDiscretizer(n_bins=5, encode="ordinal")},
)
```

### Both Together

```python
X_train, X_test, y_train, y_test = dc.prepare(
    custom_encoders={"city": OrdinalEncoder(), "gender": LabelEncoder()},
    custom_scalers={"age": RobustScaler(), "salary": StandardScaler()},
)
```

These transformers are stored in `dc.pipeline.custom_encoders` and `dc.pipeline.custom_scalers` and are applied during `transform()` as well.

---

## Imbalanced Data Handling (SMOTE)

For classification problems with imbalanced classes:

```python
X_train, X_test, y_train, y_test = dc.prepare(handle_imbalance=True)
```

SMOTE is applied to the training set after the train/test split. Requires `imbalanced-learn`:

```bash
pip install imbalanced-learn
```

---

## Saving and Loading the Pipeline

### Saving

```python
dc.save_pipeline("my_pipeline.pkl")
```

The pipeline stores everything: scalers, encoders, imputer, column lists, thresholds, etc.

### Loading

```python
dc = DataCleaner.load_pipeline("my_pipeline.pkl")
```

### Helper Methods

```python
# Get the raw pipeline object
pipeline = dc.get_pipeline()

# Direct save/load via CleanPipeline
pipeline.save("pipe.pkl")
loaded = CleanPipeline.load("pipe.pkl")
```

---

## Inference on New Data

Two approaches:

### Approach 1: Via DataCleaner

```python
dc = DataCleaner.load_pipeline("my_pipeline.pkl")
new_data = pd.DataFrame({
    "age": [28, 42],
    "salary": [65000, 95000],
    "city": ["Tehran", "Isfahan"],
    "gender": ["F", "M"],
})
processed = dc.transform(new_data)
predictions = model.predict(processed)
```

### Approach 2: Via CleanPipeline Directly

```python
pipeline = CleanPipeline.load("my_pipeline.pkl")
processed = pipeline.transform(new_data)
```

**Important**: If a column is missing from the new data, it is filled with 0. The output column order matches the training set exactly.

---

## Data Profiling Report

Generates a self-contained HTML report:

```python
dc.summary()         # returns a dict
dc.profile_report()  # returns HTML string
dc.profile_report("report.html")  # saves to file
```

The report includes:
- Overview cards (rows, columns, nulls, duplicates, memory)
- Data quality warnings
- Column details table (type, nulls, quantiles, skew, outliers)
- Numeric distribution histograms (requires matplotlib)
- Correlation heatmap
- Categorical bar charts

*Requires `matplotlib` for images, but the HTML works without it (text-only mode).*

---

## Schema Validation

Checks that required columns exist and have the expected dtypes:

```python
issues = dc.validate_schema(
    expected_schema={
        "age": "numeric",
        "date": "datetime",
        "name": "string",
    },
    required_cols=["age", "salary", "city"],
)

if issues:
    print("Validation issues found:")
    for issue in issues:
        print(f"- {issue}")
else:
    print("Schema validation passed")
```

---

## Auto-Fix Data Types

Automatically converts object columns to datetime or numeric:

```python
fixes = dc.auto_fix_dtypes()
# Result: ["date_col: object -> datetime", "price_col: object -> numeric"]
```

How it works:
1. First tries datetime parsing (>70% success rate)
2. If that fails, tries numeric conversion (supports comma thousands separator)

---

## Removing Duplicate Rows

```python
# Remove duplicates based on all columns
dc.drop_duplicates()

# Remove based on specific columns
dc.drop_duplicates(subset=["age", "city"])

# Keep the last occurrence
dc.drop_duplicates(keep="last")
```

---

## Exporting Cleaned Data

```python
# Features only
dc.export_cleaned("cleaned_data.csv")

# With target column
dc.export_cleaned("cleaned_with_target.xlsx", include_target=True)
```

Supported formats: CSV and Excel (.xlsx)

---

## Statistical Tests (Stats Module)

The `data_cleaner.stats` module provides 20+ standalone statistical functions and a `StatisticalTestSuite` class.

### Standalone Functions

```python
from data_cleaner import stats
import pandas as pd

# Normality test
result = stats.normality_test(series, method="shapiro")
# Returns: {"statistic": ..., "p_value": ..., "is_normal": True/False}

# Correlation test
result = stats.correlation_test(x, y, method="pearson")

# Chi-square test of independence
result = stats.chi_square_test(col_a, col_b)

# One-way ANOVA
result = stats.anova_one_way(group1, group2, group3)

# One-sample t-test
result = stats.t_test_one_sample(series, pop_mean=0)

# Independent t-test
result = stats.t_test_independent(a, b)

# Paired t-test
result = stats.t_test_paired(before, after)

# One-sample z-test
result = stats.z_test_one_sample(series, pop_mean=100)

# Two-sample z-test
result = stats.z_test_two_sample(a, b)

# One-sample proportion z-test
result = stats.z_test_proportion(successes=45, n=100, p_pop=0.5)

# Two-sample proportion z-test
result = stats.z_test_two_proportion(s1=30, n1=100, s2=50, n2=100)

# Variance equality test
result = stats.variance_test(a, b, method="levene")

# Kolmogorov-Smirnov test
result = stats.ks_test(series_a, series_b)

# Mutual Information
result = stats.mutual_information(X, y)

# AB test (means)
result = stats.ab_test_mean(control_series, treatment_series)

# AB test (proportions)
result = stats.ab_test_proportion(control_series, treatment_series)
```

### StatisticalTestSuite

This class integrates with `DataCleaner` and runs tests on the loaded data:

```python
from data_cleaner import DataCleaner, stats

dc = DataCleaner()
dc.load_df(data).set_target("purchased")

suite = stats.StatisticalTestSuite(dc)

# Test normality of all numeric columns
suite.test_normality()

# Correlations with target
suite.test_correlations(target_col="purchased")

# Chi-square between categorical columns
suite.test_chi_square("gender", "city")

# ANOVA
suite.test_anova("age", "city")

# Independent t-test
suite.test_t_independent("age", "score")

# AB test by group
suite.test_ab_by_group(
    metric_col="converted",
    group_col="group",
    control_value="A",
    treatment_value="B",
    metric_type="proportion",
)

# Display all results
print(suite.summary())
```

Sample `summary()` output:

```
=== Normality Tests ===
  age: NORMAL (p=0.2341)
  salary: NOT NORMAL (p=0.0012)

=== Mutual Information ===
  age: 0.1234
  salary: 0.0891
  city: 0.0456

=== AB Test ===
  Significant: YES (p=0.0031)
  Lift: 12.5%
  CI: (0.0234, 0.0891)
  Control: 0.32 (n=500)
  Treatment: 0.45 (n=480)
```

---

## Visualization Module (Plotting)

The optional `data_cleaner.plotting` module requires `matplotlib` and `seaborn`:

```bash
pip install data-cleaner[plot]
```

```python
from data_cleaner import plotting

# Null value report (bar chart)
plotting.plot_null_report(dc)

# Distribution of numeric columns (histogram + boxplot)
plotting.plot_distributions(dc)
plotting.plot_distributions(dc, cols=["age", "salary"])

# Correlation heatmap
plotting.plot_correlation(dc)
plotting.plot_correlation(dc, figsize=(12, 10))

# Before/After distribution comparison
plotting.plot_before_after(dc)  # requires .prepare() to have been called
```

---

## Complete End-to-End Example

### Step 1: Train and Save Pipeline

```python
from data_cleaner import DataCleaner
import pandas as pd
from sklearn.svm import SVC
import joblib

# Create sample data
data = pd.DataFrame({
    "ID": range(100),
    "age": [25, 30, 35, None, 40, 45, 50, 55, 60, 65] * 10,
    "salary": [50000, 60000, None, 80000, 90000, 100000, 110000, 120000, None, 140000] * 10,
    "city": ["Tehran", "Shiraz", "Tehran", "Isfahan", None, "Tehran", "Shiraz", "Isfahan", "Tehran", "Shiraz"] * 10,
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"] * 10,
    "purchased": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 10,
    "register_date": pd.date_range("2024-01-01", periods=100, freq="D"),
})

# Load and configure
dc = DataCleaner(random_state=42)
dc.load_df(data)
dc.set_target("purchased")
dc.drop_columns(["ID"])

# Quick summary
info = dc.summary()
print(f"Shape: {info['shape']}")
print(f"Nulls: {info['null_counts']}")

# Run the full pipeline with all features
X_train, X_test, y_train, y_test = dc.prepare(
    test_size=0.2,
    handle_nulls=True,
    auto_scale=True,
    auto_encode=True,
    auto_drop_useless=True,
    handle_outliers="clip",
    feature_engineering=True,
    extract_date_features=True,
    add_missing_indicators=True,
    feature_selection="auto",
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {list(X_train.columns)}")
print(f"Scalers: {list(dc.pipeline.scalers.keys())}")
print(f"Problem type: {dc.pipeline.problem_type}")
print(f"Useless dropped: {dc.pipeline.dropped_useless_cols}")
print(f"Feature selection dropped: {dc.pipeline.feature_selection_removed}")

# Train model
model = SVC(probability=True)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")

# Test pipeline on raw data
new_data = pd.DataFrame({
    "age": [28, 42, 35],
    "salary": [65000, 95000, 78000],
    "city": ["Tehran", "Isfahan", "Shiraz"],
    "gender": ["F", "M", "F"],
    "register_date": pd.to_datetime(["2024-06-15", "2024-07-20", "2024-08-10"]),
})
processed = dc.transform(new_data)
predictions = model.predict(processed)
probabilities = model.predict_proba(processed)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "Purchased" if pred == 1 else "Not Purchased"
    print(f"Customer {i+1}: {status} (confidence: {max(prob):.2%})")

# Export cleaned data
dc.export_cleaned("cleaned_data.csv")
dc.export_cleaned("cleaned_with_target.xlsx", include_target=True)

# Save everything
joblib.dump(model, "model.pkl")
dc.save_pipeline("full_pipeline.pkl")
```

### Step 2: Inference on New Data

```python
from data_cleaner import DataCleaner
import pandas as pd
from sklearn.svm import SVC
import joblib

# Load pipeline and model
dc = DataCleaner.load_pipeline("full_pipeline.pkl")
model = joblib.load("model.pkl")

# New customer data
new_customers = pd.DataFrame({
    "age": [28, 42, 35],
    "salary": [65000, 95000, 78000],
    "city": ["Tehran", "Isfahan", "Shiraz"],
    "gender": ["F", "M", "F"],
    "register_date": pd.to_datetime(["2024-06-15", "2024-07-20", "2024-08-10"]),
})

# Transform and predict
processed = dc.transform(new_customers)
predictions = model.predict(processed)
probabilities = model.predict_proba(processed)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    status = "Purchased" if pred == 1 else "Not Purchased"
    print(f"Customer {i+1}: {status} (confidence: {max(prob):.2%})")
```

### Step 3: Statistical Analysis

```python
from data_cleaner import DataCleaner, stats

dc = DataCleaner()
dc.load_df(data)
dc.set_target("purchased")

suite = stats.StatisticalTestSuite(dc)

# Run various tests
normality = suite.test_normality()
correlations = suite.test_correlations(target_col="purchased")
chi2 = suite.test_chi_square("gender", "city")
anova = suite.test_anova("age", "city")
ztest = suite.test_z_one_sample("age", pop_mean=35)
ab_result = suite.test_ab_by_group(
    "purchased", "gender", "M", "F", metric_type="proportion"
)
mutual_info = suite.test_mutual_info()

# View summary
print(suite.summary())
```

---

## Tips and Best Practices

### 1. Tree-Based Models Don't Need Scaling

Tree-based models (Random Forest, XGBoost, LightGBM, Decision Tree, Gradient Boosting) do not require normalization or scaling. Use `auto_scale=False`:

```python
X_train, X_test, y_train, y_test = dc.prepare(auto_scale=False)
```

### 2. Method Chaining

All configuration methods return `self`, enabling fluent chaining:

```python
dc = DataCleaner()
dc.load_df(data).set_target("purchased").drop_columns(["ID"])
```

### 3. Call set_target() Before prepare()

If the target is not set, `prepare()` raises an error:

```
ValueError: Target column not set. Use .set_target() first.
```

### 4. Save the Pipeline, Not the DataCleaner

Use `dc.save_pipeline()` rather than `pickle.dump(dc)`. The pipeline is lighter and contains only what is needed for inference.

### 5. Inference Data Should Not Include the Target

New data for inference should not have the target column. If it does, `transform()` will keep it (it's in `feature_cols`).

### 6. Use n_jobs for Speed on Large Datasets

For large datasets with many numeric columns:

```python
X_train, X_test, y_train, y_test = dc.prepare(n_jobs=-1)  # use all CPU cores
```

### 7. Combine Feature Engineering and Feature Selection

Using `feature_engineering=True` with `feature_selection="auto"` keeps the best polynomial features and drops weak ones.

### 8. Missing Indicators Help Tree Models

Enable `add_missing_indicators=True` to help models like XGBoost and Random Forest learn patterns from null locations.

### 9. Use Date Extraction for Temporal Data

For datasets with datetime columns, enable `extract_date_features=True` to get year, month, day, dayofweek, and weekend indicators.

### 10. Export Cleaned Data for Analysis

Use `export_cleaned()` to get the cleaned dataset for analysis in other tools (Excel, Tableau, Power BI).

---

## Resources

- **GitHub**: https://github.com/MohammadvHossein/data_cleaner
- **PyPI**: https://pypi.org/project/data-cleaner/
- **Author**: Mohammad Hossein Habibpour
- **Email**: habibpour.programming@gmail.com
- **License**: MIT
