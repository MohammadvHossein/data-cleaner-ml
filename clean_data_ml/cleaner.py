"""Core DataCleaner and CleanPipeline classes.

``DataCleaner`` orchestrates the full data-cleaning workflow,
while ``CleanPipeline`` holds fitted transformers for inference.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .auto_scaler import Scaler, select_best_scaler

logger = logging.getLogger(__name__)

try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


SplitResult = Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series],
]


# =============================================================================
# CleanPipeline
# =============================================================================


class CleanPipeline:
    """Holds all fitted transformers produced by ``DataCleaner.prepare()``.

    Used to transform new raw data during inference via ``.transform()``.
    Can be serialised with ``.save()`` / ``.load()``.

    Attributes
    ----------
    columns_to_drop : list of str
        Columns removed during transformation.
    target_col : str or None
        Name of the target column.
    feature_cols : list of str or None
        Ordered list of feature columns expected by the model.
    numeric_cols : list of str
        Columns treated as numeric (multi-valued).
    categorical_cols : list of str
        Columns one-hot encoded during training.
    binary_cols : list of str
        Columns label-encoded (2 unique values).
    scalers : dict of str -> Scaler
        Column -> fitted sklearn scaler.
    label_encoders : dict of str -> dict
        Column -> {class: encoded_value} mapping.
    onehot_encoder : OneHotEncoder or None
        Fitted one-hot encoder.
    onehot_cols : list of str
        Output column names from one-hot encoding.
    imputer : KNNImputer or None
        Fitted KNN imputer for numeric columns.
    impute_cols : list of str
        Numeric columns that were KNN-imputed.
    cat_impute_values : dict of str -> any
        Column -> mode value for categorical imputation.
    used_knn : bool
        Whether KNN imputation was applied.
    problem_type : str or None
        ``"classification"`` or ``"regression"``.
    dropped_useless_cols : list of str
        Columns auto-dropped by ``auto_drop_useless``.
    outlier_bounds : dict of str -> dict
        Column -> ``{"lower": float, "upper": float}``.
    poly_features : PolynomialFeatures or None
        Fitted polynomial feature transformer.
    poly_numeric_cols : list of str
        Numeric columns used as input to ``poly_features``.
    date_cols : list of str
        Datetime columns expanded into year/month/day/etc.
    date_feature_cols : list of str
        Names of the extracted date feature columns.
    missing_indicator_cols : list of str
        Original columns for which ``{col}_missing`` indicators were added.
    feature_selection_threshold : float or None
        MI threshold below which features were removed.
    feature_selection_removed : list of str
        Columns removed by feature selection.
    custom_encoders : dict of str -> encoder
        Column -> fitted custom encoder instance.
    custom_scalers : dict of str -> scaler
        Column -> fitted custom scaler instance.
    """

    def __init__(self) -> None:
        self.columns_to_drop: List[str] = []
        self.target_col: Optional[str] = None
        self.feature_cols: Optional[List[str]] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.binary_cols: List[str] = []
        self.scalers: Dict[str, Scaler] = {}
        self.label_encoders: Dict[str, Dict[Any, int]] = {}
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.onehot_cols: List[str] = []
        self.imputer: Optional[KNNImputer] = None
        self.impute_cols: List[str] = []
        self.cat_impute_values: Dict[str, Any] = {}
        self.used_knn: bool = False
        self.problem_type: Optional[str] = None
        self.dropped_useless_cols: List[str] = []
        self.outlier_bounds: Dict[str, Dict[str, float]] = {}
        self.poly_features: Any = None
        self.poly_numeric_cols: List[str] = []
        self.date_cols: List[str] = []
        self.date_feature_cols: List[str] = []
        self.missing_indicator_cols: List[str] = []
        self.feature_selection_threshold: Optional[float] = None
        self.feature_selection_removed: List[str] = []
        self.custom_encoders: Dict[str, Any] = {}
        self.custom_scalers: Dict[str, Any] = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps to a raw DataFrame.

        Order of operations:
        1. Drop configured columns
        2. KNN-impute numeric nulls
        3. Add missing indicators for imputed columns
        4. Fill categorical nulls with stored mode values
        5. Clip outliers if bounds were computed
        6. Apply custom encoders (if configured)
        7. Label-encode binary columns
        8. One-hot encode categorical columns
        9. Scale numeric columns (auto-selected scalers)
        10. Apply custom scalers (if configured)
        11. Expand datetime columns (if date_cols stored)
        12. Generate polynomial features (if configured)
        13. Zero-fill any missing expected columns
        14. Return columns in the order of ``feature_cols``

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data (can be a subset of training columns).

        Returns
        -------
        pd.DataFrame
            Transformed data with the same column order as training.
        """
        df = df.copy()

        for col in self.columns_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        if self.imputer is not None:
            present = [c for c in self.impute_cols if c in df.columns]
            if present:
                df[present] = self.imputer.transform(df[present])

        if self.missing_indicator_cols:
            for col in self.missing_indicator_cols:
                if col in df.columns:
                    df[f"{col}_missing"] = df[col].isnull().astype(int)

        for col in self.categorical_cols:
            if col in df.columns:
                fill_val = self.cat_impute_values.get(col)
                df[col] = df[col].fillna(fill_val if fill_val is not None else "unknown")
                df[col] = df[col].astype(str)

        if self.outlier_bounds:
            for col, bounds in self.outlier_bounds.items():
                if col in df.columns:
                    df[col] = df[col].clip(bounds["lower"], bounds["upper"])

        if self.custom_encoders:
            for col, encoder in self.custom_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform(df[[col]] if hasattr(encoder, "transform") else df[col])

        for col in self.binary_cols:
            if col in df.columns:
                mapping = self.label_encoders.get(col, {})
                fallback = next(iter(mapping.keys())) if mapping else "unknown"
                df[col] = df[col].fillna(fallback).map(mapping).fillna(0)

        if self.onehot_encoder is not None and self.categorical_cols:
            present = [c for c in self.categorical_cols if c in df.columns]
            if present:
                encoded = self.onehot_encoder.transform(df[present])
                encoded_df = pd.DataFrame(encoded, columns=self.onehot_cols, index=df.index)
                df = pd.concat([df.drop(columns=present), encoded_df], axis=1)

        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])

        if self.custom_scalers:
            for col, scaler in self.custom_scalers.items():
                if col in df.columns:
                    df[col] = scaler.transform(df[[col]])

        if self.date_cols:
            for col in self.date_cols:
                if col in df.columns:
                    dt = df[col].dt
                    df[f"{col}_year"] = dt.year
                    df[f"{col}_month"] = dt.month
                    df[f"{col}_day"] = dt.day
                    df[f"{col}_dayofweek"] = dt.dayofweek
                    df[f"{col}_weekend"] = (dt.dayofweek >= 5).astype(int)
                    df.drop(columns=[col], inplace=True)

        if self.poly_features is not None and self.poly_numeric_cols:
            present = [c for c in self.poly_numeric_cols if c in df.columns]
            if len(present) >= 2:
                missing_poly = set(self.poly_numeric_cols) - set(present)
                if missing_poly:
                    for col in missing_poly:
                        df[col] = 0
                    present = self.poly_numeric_cols
                out = self.poly_features.transform(df[present])
                names = self.poly_features.get_feature_names_out(present).tolist()
                new_names = [n for n in names if n not in df.columns]
                poly_df = pd.DataFrame(out[:, len(present):], columns=new_names, index=df.index)
                df = pd.concat([df, poly_df], axis=1)

        missing = set(self.feature_cols or []) - set(df.columns)
        for col in missing:
            df[col] = 0

        return df[self.feature_cols] if self.feature_cols else df

    def save(self, path: str) -> None:
        """Persist the pipeline to disk via pickle.

        Parameters
        ----------
        path : str
            File path (e.g. ``"pipeline.pkl"``).
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> CleanPipeline:
        """Load a previously saved pipeline from disk.

        .. warning::
            ``pickle.load`` can execute arbitrary code. Only load
            pipelines from trusted sources.

        Parameters
        ----------
        path : str
            File path to the saved pipeline.

        Returns
        -------
        CleanPipeline
            Deserialised pipeline instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


# =============================================================================
# DataCleaner
# =============================================================================


class DataCleaner:
    """End-to-end data cleaning and ML pipeline preparation.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility (default 42).

    Attributes
    ----------
    df : pd.DataFrame or None
        Current working DataFrame after loading.
    raw_df : pd.DataFrame or None
        Snapshot of the original data before any modifications.
    target_col : str or None
        Name of the designated target column.
    columns_to_drop : list of str
        Columns marked for removal via ``.drop_columns()``.
    pipeline : CleanPipeline
        The fitted transformation pipeline, populated after ``.prepare()``.
        Contains all fitted transformers, scalers, encoders, etc.
    X_train, X_test, y_train, y_test : pd.DataFrame/Series or None
        Train/test splits produced by ``.prepare()``.
    X_val, y_val : pd.DataFrame/Series or None
        Optional validation split (if ``val_size`` was set in ``.prepare()``).
    X_clean, y_clean : pd.DataFrame/Series or None
        Fully cleaned data before splitting (available after ``.prepare()``).
    is_fitted : bool
        Whether ``.prepare()`` has been successfully called.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.target_col: Optional[str] = None
        self.columns_to_drop: List[str] = []
        self.pipeline = CleanPipeline()
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None
        self.X_clean: Optional[pd.DataFrame] = None
        self.y_clean: Optional[pd.Series] = None
        self.is_fitted: bool = False

    # -- I/O -----------------------------------------------------------------

    def load(self, filepath: str) -> DataCleaner:
        """Load data from a CSV or Excel file.

        Parameters
        ----------
        filepath : str
            Path to ``.csv`` or ``.xlsx`` file.

        Returns
        -------
        DataCleaner
            Self, for method chaining.
        """
        if filepath.endswith(".csv"):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith((".xls", ".xlsx")):
            self.df = pd.read_excel(filepath)
        else:
            raise ValueError("Only CSV and Excel files are supported")
        self.raw_df = self.df.copy()
        return self

    def load_df(self, dataframe: pd.DataFrame) -> DataCleaner:
        """Load data from an in-memory pandas DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Existing DataFrame to clean.

        Returns
        -------
        DataCleaner
            Self, for method chaining.
        """
        self.df = dataframe.copy()
        self.raw_df = self.df.copy()
        return self

    # -- Configuration -------------------------------------------------------

    def set_target(self, column: str) -> DataCleaner:
        """Designate a column as the prediction target.

        Parameters
        ----------
        column : str
            Name of the target column.

        Returns
        -------
        DataCleaner
            Self, for method chaining.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in data")
        self.target_col = column
        return self

    def drop_columns(self, columns: List[str]) -> DataCleaner:
        """Mark columns to be dropped before cleaning.

        Safe to call multiple times — columns are appended, not replaced.

        Parameters
        ----------
        columns : list of str
            Column names to drop.

        Returns
        -------
        DataCleaner
            Self, for method chaining.
        """
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in data")
        for col in columns:
            if col not in self.columns_to_drop:
                self.columns_to_drop.append(col)
        return self

    # -- Internal helpers ----------------------------------------------------

    def _auto_detect_types(
        self, X: Optional[pd.DataFrame] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """Categorise columns into numeric, categorical, and binary.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature DataFrame to analyse. If None, uses ``self.df``
            after applying column drops.

        Returns
        -------
        tuple of (numeric_cols, categorical_cols, binary_cols)
        """
        if X is not None:
            df = X
        else:
            df = self.df.copy()
            for col in self.columns_to_drop:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            if self.target_col and self.target_col in df.columns:
                df.drop(columns=[self.target_col], inplace=True)

        numeric, categorical, binary = [], [], []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                (binary if df[col].nunique() == 2 else numeric).append(col)
            else:
                (binary if df[col].nunique() == 2 else categorical).append(col)

        return numeric, categorical, binary

    def _select_scalers(
        self, numeric_cols: List[str], df: pd.DataFrame, n_jobs: int = 1
    ) -> Dict[str, Scaler]:
        """Select the best scaler for each numeric column, optionally in parallel."""
        if n_jobs != 1 and _HAS_JOBLIB and len(numeric_cols) > 5:
            results = Parallel(n_jobs=n_jobs)(
                delayed(select_best_scaler)(df[col]) for col in numeric_cols
            )
            return dict(zip(numeric_cols, results))
        return {col: select_best_scaler(df[col]) for col in numeric_cols}

    @staticmethod
    def _calc_dynamic_threshold(n_rows: int) -> float:
        """Compute a dynamic null-drop threshold based on dataset size.

        Larger datasets can tolerate more aggressive row dropping.
        See :ref:`How Nulls Are Handled` in the README.
        """
        threshold = 0.05 * (1000 / max(n_rows, 1))
        return round(max(0.01, min(0.25, threshold)), 4)

    def _handle_nulls(
        self, df: pd.DataFrame, max_drop_ratio: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], List[str]]:
        """Decide which columns to drop rows from vs impute.

        Columns with null ratio <= threshold have their null rows dropped;
        columns with higher null ratio are marked for imputation.

        Returns
        -------
        (cleaned_df, null_info, impute_cols)
        """
        df = df.copy()
        if max_drop_ratio is None:
            max_drop_ratio = self._calc_dynamic_threshold(len(df))

        null_info: Dict[str, Dict[str, float]] = {}
        impute_cols: List[str] = []
        drop_cols: List[str] = []

        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count == 0:
                continue
            null_ratio = null_count / len(df)
            null_info[col] = {"count": null_count, "ratio": null_ratio}
            (drop_cols if null_ratio <= max_drop_ratio else impute_cols).append(col)

        if drop_cols:
            df = df.dropna(subset=drop_cols)

        return df, null_info, impute_cols

    def _impute_nulls(self, df: pd.DataFrame, impute_cols: List[str]) -> pd.DataFrame:
        """Impute missing values: KNN for numeric, mode fill for categorical.

        Stores fitted imputer and categorical impute values in ``self.pipeline``.
        """
        if not impute_cols:
            return df

        numeric = [c for c in impute_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in impute_cols if not pd.api.types.is_numeric_dtype(df[c])]

        if numeric:
            knn = KNNImputer(n_neighbors=5)
            df[numeric] = knn.fit_transform(df[numeric])
            self.pipeline.imputer = knn
            self.pipeline.impute_cols = numeric
            self.pipeline.used_knn = True

        for col in categorical:
            mode_series = df[col].mode()
            fill_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
            df[col] = df[col].fillna(fill_val)
            self.pipeline.cat_impute_values[col] = fill_val

        return df

    @staticmethod
    def _detect_problem_type(y: pd.Series) -> str:
        """Detect classification vs regression from the target series."""
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        uniq = y.nunique()
        if uniq > 25:
            return "regression"
        if uniq < 10:
            return "classification"
        ratios = y.value_counts(normalize=True)
        return "classification" if ratios.max() > 0.5 else "regression"

    def _auto_drop_useless(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop zero-variance and high-cardinality non-numeric columns."""
        drop_cols: set = set()
        n = len(X)

        for col in X.columns:
            if X[col].nunique() <= 1:
                drop_cols.add(col)
            elif not pd.api.types.is_numeric_dtype(X[col]) and n > 10:
                if X[col].nunique() / n > 0.9:
                    drop_cols.add(col)

        X = X.drop(columns=list(drop_cols))
        self.pipeline.dropped_useless_cols = list(drop_cols)
        return X

    def _handle_outliers(
        self, X: pd.DataFrame, method: str = "clip", n_jobs: int = 1
    ) -> pd.DataFrame:
        """Detect and handle outliers using the IQR method.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        method : {"clip", "remove"}
            ``"clip"`` caps outliers at bounds; ``"remove"`` drops outlier rows.
        n_jobs : int
            Parallel workers for bound computation.

        Returns
        -------
        pd.DataFrame
            Data with outliers handled (may have fewer rows if ``method="remove"``).
        """
        if n_jobs != 1 and _HAS_JOBLIB:
            cols = [
                c for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 2
            ]
            if not cols:
                return X

            def _bounds(col: str) -> Tuple[str, float, float]:
                q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                iqr = q3 - q1
                return col, float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)

            results = Parallel(n_jobs=n_jobs)(delayed(_bounds)(c) for c in cols)
            bounds = {col: {"lower": lo, "upper": hi} for col, lo, hi in results}
            self.pipeline.outlier_bounds = bounds
            for col, b in bounds.items():
                if method == "clip":
                    X[col] = X[col].clip(b["lower"], b["upper"])
                elif method == "remove":
                    X = X[(X[col] >= b["lower"]) & (X[col] <= b["upper"])]
            return X

        bounds: Dict[str, Dict[str, float]] = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]) or X[col].nunique() <= 2:
                continue
            q1, q3 = float(X[col].quantile(0.25)), float(X[col].quantile(0.75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            bounds[col] = {"lower": lo, "upper": hi}
            if method == "clip":
                X[col] = X[col].clip(lo, hi)
            elif method == "remove":
                X = X[(X[col] >= lo) & (X[col] <= hi)]
        self.pipeline.outlier_bounds = bounds
        return X

    def _feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features (degree 2) for numeric columns."""
        from sklearn.preprocessing import PolynomialFeatures

        poly_cols = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 2
        ]
        if len(poly_cols) < 2:
            return X

        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        out = poly.fit_transform(X[poly_cols])
        names = poly.get_feature_names_out(poly_cols).tolist()
        new_names = [n for n in names if n not in set(X.columns)]

        poly_df = pd.DataFrame(out[:, len(poly_cols):], columns=new_names, index=X.index)
        self.pipeline.poly_features = poly
        self.pipeline.poly_numeric_cols = poly_cols
        return pd.concat([X, poly_df], axis=1)

    def _extract_date_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Expand datetime columns into year/month/day/dayofweek/is_weekend.

        Detects columns with ``datetime64`` dtype, extracts components,
        and drops the original date column.

        Returns
        -------
        pd.DataFrame
            Feature matrix with date columns expanded.
        """
        date_cols = [
            c for c in X.columns
            if pd.api.types.is_datetime64_dtype(X[c])
        ]
        if not date_cols:
            return X

        self.pipeline.date_cols = date_cols
        date_feature_cols = []
        for col in date_cols:
            dt = X[col].dt
            X[f"{col}_year"] = dt.year
            X[f"{col}_month"] = dt.month
            X[f"{col}_day"] = dt.day
            X[f"{col}_dayofweek"] = dt.dayofweek
            X[f"{col}_weekend"] = (dt.dayofweek >= 5).astype(int)
            date_feature_cols.extend([
                f"{col}_year", f"{col}_month", f"{col}_day",
                f"{col}_dayofweek", f"{col}_weekend",
            ])
            X.drop(columns=[col], inplace=True)
        self.pipeline.date_feature_cols = date_feature_cols
        logger.info(f"Extracted date features: {date_feature_cols}")
        return X

    def _add_missing_indicators(
        self, X: pd.DataFrame, impute_cols: List[str]
    ) -> pd.DataFrame:
        """Add binary ``{col}_missing`` columns for imputed columns.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (after null row dropping).
        impute_cols : list of str
            Columns that will be imputed (had null ratio above threshold).

        Returns
        -------
        pd.DataFrame
            Feature matrix with missing indicator columns appended.
        """
        for col in impute_cols:
            if col in X.columns:
                X[f"{col}_missing"] = X[col].isnull().astype(int)
        self.pipeline.missing_indicator_cols = impute_cols.copy()
        if impute_cols:
            logger.info(f"Added missing indicators for: {impute_cols}")
        return X

    def _feature_selection(
        self, X: pd.DataFrame, y: pd.Series,
        threshold: Union[str, float] = "auto",
    ) -> pd.DataFrame:
        """Remove weak features using mutual information.

        Parameters
        ----------
        X : pd.DataFrame
            Fully transformed feature matrix.
        y : pd.Series
            Target vector.
        threshold : str or float
            ``"auto"`` uses median MI; a float removes features with MI below it.

        Returns
        -------
        pd.DataFrame
            Feature matrix with weak columns removed.
        """
        numeric = X.select_dtypes(include="number").columns.tolist()
        if not numeric or y.nunique() <= 1:
            return X

        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        is_classification = y.nunique() < 20
        fn = mutual_info_classif if is_classification else mutual_info_regression
        mi = fn(X[numeric], y, random_state=self.random_state)
        mi_series = pd.Series(mi, index=numeric)

        actual_threshold = mi_series.median() if threshold == "auto" else float(threshold)
        self.pipeline.feature_selection_threshold = actual_threshold

        to_drop = mi_series[mi_series < actual_threshold].index.tolist()
        if to_drop:
            X = X.drop(columns=to_drop)
            self.pipeline.feature_selection_removed = to_drop
            logger.info(f"Feature selection dropped {len(to_drop)} weak columns: {to_drop}")
        return X

    def _apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE oversampling if imbalanced-learn is available."""
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE applied. New train size: {len(X_train)}")
        except ImportError:
            logger.warning("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
        return X_train, y_train

    def _encode_binary(self, df: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
        """Label-encode binary columns (2 unique values)."""
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.pipeline.label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        return df

    def _encode_categorical(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """One-hot encode multi-category columns."""
        if not categorical_cols:
            return df

        for col in categorical_cols:
            mode_series = df[col].mode()
            fill_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
            df[col] = df[col].fillna(fill_val).astype(str)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(df[categorical_cols])
        self.pipeline.onehot_encoder = ohe
        self.pipeline.onehot_cols = ohe.get_feature_names_out(categorical_cols).tolist()

        encoded_df = pd.DataFrame(encoded, columns=self.pipeline.onehot_cols, index=df.index)
        return pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    def _scale_features(self, df: pd.DataFrame, scalers: Dict[str, Scaler]) -> pd.DataFrame:
        """Apply fitted scalers to numeric columns."""
        for col, scaler in scalers.items():
            df[col] = scaler.fit_transform(df[[col]])
        return df

    def _split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float, val_size: Optional[float]
    ) -> SplitResult:
        """Split data into train/test (and optional validation) sets."""
        if val_size is not None:
            test_val = test_size + val_size
            X_tr, X_tv, y_tr, y_tv = train_test_split(
                X, y, test_size=test_val, random_state=self.random_state
            )
            val_ratio = val_size / test_val if test_val > 0 else 0
            X_v, X_te, y_v, y_te = train_test_split(
                X_tv, y_tv, test_size=1 - val_ratio, random_state=self.random_state
            )
            self.X_train, self.X_val, self.X_test = X_tr, X_v, X_te
            self.y_train, self.y_val, self.y_test = y_tr, y_v, y_te
            self.is_fitted = True
            return X_tr, X_v, X_te, y_tr, y_v, y_te

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te
        self.is_fitted = True
        return X_tr, X_te, y_tr, y_te

    # -- Main pipeline -------------------------------------------------------

    def prepare(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        handle_nulls: bool = True,
        auto_scale: bool = True,
        auto_encode: bool = True,
        null_drop_ratio: Optional[float] = None,
        auto_drop_useless: bool = True,
        handle_outliers: Optional[str] = None,
        feature_engineering: bool = False,
        handle_imbalance: bool = False,
        n_jobs: int = 1,
        extract_date_features: bool = False,
        add_missing_indicators: bool = False,
        feature_selection: Optional[Union[str, float]] = None,
        custom_encoders: Optional[Dict[str, Any]] = None,
        custom_scalers: Optional[Dict[str, Any]] = None,
    ) -> SplitResult:
        """Execute the full cleaning and preparation pipeline.

        Pipeline order:
            drop columns -> detect problem type -> extract date features ->
            auto-drop useless -> detect types -> handle nulls ->
            add missing indicators -> handle outliers -> feature engineering ->
            encode -> scale -> feature selection -> split -> optional SMOTE

        Parameters
        ----------
        test_size : float
            Fraction of data for testing (default 0.2).
        val_size : float, optional
            If set, also creates a validation set.
        handle_nulls : bool
            Auto-detect and handle missing values (default True).
        auto_scale : bool
            Auto-select and apply optimal scaler per column (default True).
        auto_encode : bool
            Auto-encode binary (Label) and categorical (OneHot) columns (default True).
        null_drop_ratio : float, optional
            Override the dynamic null threshold.
        auto_drop_useless : bool
            Drop zero-variance and high-cardinality columns (default True).
        handle_outliers : {"clip", "remove"}, optional
            Outlier handling strategy. ``None`` skips.
        feature_engineering : bool
            Add polynomial features (default False).
        handle_imbalance : bool
            Apply SMOTE on imbalanced classification data (default False).
        n_jobs : int
            Parallel workers for scaler selection and outlier handling.
        extract_date_features : bool
            Expand datetime columns into year/month/day/dayofweek/weekend (default False).
        add_missing_indicators : bool
            Add ``{col}_missing`` binary columns for imputed columns (default False).
        feature_selection : str or float, optional
            ``"auto"`` drops features below median mutual information;
            a float threshold drops features below that value.
            ``None`` skips feature selection.
        custom_encoders : dict of str -> encoder, optional
            Column -> sklearn-compatible encoder instance. Overrides auto-encoding
            for the specified columns.
        custom_scalers : dict of str -> scaler, optional
            Column -> sklearn-compatible scaler instance. Overrides auto-scaling
            for the specified columns.

        Returns
        -------
        tuple of DataFrames/Series
            ``(X_train, X_test, y_train, y_test)`` or
            ``(X_train, X_val, X_test, y_train, y_val, y_test)`` if ``val_size`` is set.
        """
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")
        df = self.df.copy()

        if self.target_col is None:
            raise ValueError("Target column not set. Use .set_target() first.")

        self.pipeline.columns_to_drop = self.columns_to_drop
        self.pipeline.target_col = self.target_col

        for col in self.columns_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])
        self.pipeline.problem_type = self._detect_problem_type(y)

        if extract_date_features:
            X = self._extract_date_features(X)

        if auto_drop_useless:
            X = self._auto_drop_useless(X)

        numeric_cols, categorical_cols, binary_cols = self._auto_detect_types(X)
        self.pipeline.numeric_cols = numeric_cols
        self.pipeline.categorical_cols = categorical_cols
        self.pipeline.binary_cols = binary_cols

        if handle_nulls:
            y_nulls = int(y.isnull().sum())
            if y_nulls > 0:
                y_null_ratio = y_nulls / len(y)
                if y_null_ratio <= 0.05:
                    valid = y.dropna().index
                    X, y = X.loc[valid], y.loc[valid]
                else:
                    raise ValueError(
                        f"Target column '{self.target_col}' has {y_null_ratio:.1%} null "
                        "values. Please clean it manually."
                    )

            merged = pd.concat([X, y], axis=1)
            merged, _, impute_cols = self._handle_nulls(merged, null_drop_ratio)
            X = merged.drop(columns=[self.target_col])
            y = merged[self.target_col]

            if add_missing_indicators and impute_cols:
                X = self._add_missing_indicators(X, impute_cols)

            if impute_cols:
                X = self._impute_nulls(X, impute_cols)

        if handle_outliers in ("clip", "remove"):
            X = self._handle_outliers(X, method=handle_outliers, n_jobs=n_jobs)
            y = y.loc[X.index]

        if feature_engineering:
            X = self._feature_engineering(X)

        if custom_encoders:
            for col, encoder in custom_encoders.items():
                if col in X.columns:
                    X[col] = encoder.fit_transform(X[[col]] if hasattr(encoder, "fit_transform") else X[col])
                    self.pipeline.custom_encoders[col] = encoder
            encoder_cols = set(custom_encoders.keys())
            binary_cols = [c for c in binary_cols if c not in encoder_cols]
            categorical_cols = [c for c in categorical_cols if c not in encoder_cols]

        if auto_encode:
            X = self._encode_binary(X, binary_cols)
            X = self._encode_categorical(X, categorical_cols)
        else:
            for col in binary_cols + categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        if auto_scale:
            scalers = self._select_scalers(numeric_cols, X, n_jobs=n_jobs)
            self.pipeline.scalers = scalers
            X = self._scale_features(X, scalers)

        if custom_scalers:
            for col, scaler in custom_scalers.items():
                if col in X.columns:
                    X[col] = scaler.fit_transform(X[[col]])
                    self.pipeline.custom_scalers[col] = scaler

        if feature_selection is not None:
            X = self._feature_selection(X, y, threshold=feature_selection)

        self.X_clean = X
        self.y_clean = y
        self.pipeline.feature_cols = X.columns.tolist()

        if handle_imbalance and self.pipeline.problem_type == "classification":
            result = self._split_data(X, y, test_size, val_size)
            if val_size is not None:
                X_tr, X_v, X_te, y_tr, y_v, y_te = result
            else:
                X_tr, X_te, y_tr, y_te = result
            X_tr, y_tr = self._apply_smote(X_tr, y_tr)
            self.X_train, self.y_train = X_tr, y_tr
            if val_size is not None:
                return X_tr, X_v, X_te, y_tr, y_v, y_te
            return X_tr, X_te, y_tr, y_te

        return self._split_data(X, y, test_size, val_size)

    # -- Post-training -------------------------------------------------------

    def get_pipeline(self) -> CleanPipeline:
        """Return the fitted ``CleanPipeline``.

        Raises
        ------
        RuntimeError
            If ``.prepare()`` has not been called.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call .prepare() first")
        return self.pipeline

    def save_pipeline(self, path: str) -> None:
        """Save the fitted pipeline to disk.

        Parameters
        ----------
        path : str
            Output file path.
        """
        self.get_pipeline().save(path)

    @staticmethod
    def load_pipeline(path: str) -> DataCleaner:
        """Load a saved pipeline and wrap it in a ``DataCleaner``.

        The returned ``DataCleaner`` is marked as fitted and can be
        used via ``.transform()`` for inference. Note that methods
        requiring raw data (``.summary()``, ``.export_cleaned()``) will
        not be available unless new data is loaded first.

        .. warning::
            Only load pipelines from trusted sources (uses ``pickle``).

        Parameters
        ----------
        path : str
            File path to the saved pipeline.

        Returns
        -------
        DataCleaner
            A fitted ``DataCleaner`` wrapping the loaded pipeline.
        """
        pipeline = CleanPipeline.load(path)
        dc = DataCleaner()
        dc.pipeline = pipeline
        dc.target_col = pipeline.target_col
        dc.columns_to_drop = pipeline.columns_to_drop
        dc.is_fitted = True
        return dc

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline to new raw data.

        Shortcut for ``self.get_pipeline().transform(df)``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.DataFrame
            Transformed features ready for prediction.
        """
        return self.get_pipeline().transform(df)

    def export_cleaned(self, filepath: str, include_target: bool = False) -> None:
        """Export the cleaned dataset to CSV or Excel.

        Parameters
        ----------
        filepath : str
            Output path (``.csv`` or ``.xlsx``).
        include_target : bool
            Whether to include the target column (default False).
        """
        if not self.is_fitted:
            raise RuntimeError("Must call .prepare() first")
        df = self.X_clean.copy()
        if include_target and self.y_clean is not None:
            df[self.target_col] = self.y_clean.values
        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        elif filepath.endswith(".xlsx"):
            df.to_excel(filepath, index=False)
        elif filepath.endswith(".xls"):
            raise ValueError("The .xls format is not supported. Use .xlsx or .csv instead.")
        else:
            raise ValueError("Only CSV and Excel (.xlsx) formats are supported")
        logger.info(f"Cleaned data exported to {filepath}")

    def summary(self) -> Union[str, Dict[str, Any]]:
        """Return a quick overview of the loaded data.

        Returns
        -------
        dict or str
            Dict with shape, columns, dtypes, null counts/percentages,
            or ``"No data loaded"`` if no data is loaded.
        """
        if self.df is None:
            return "No data loaded"
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {str(k): str(v) for k, v in self.df.dtypes.items()},
            "null_counts": self.df.isnull().sum().to_dict(),
            "null_percent": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
        }

    def drop_duplicates(self, subset: Optional[List[str]] = None, keep: str = "first") -> DataCleaner:
        """Remove duplicate rows from the loaded data.

        Parameters
        ----------
        subset : list of str, optional
            Columns to consider for duplication. If None, all columns.
        keep : {"first", "last", False}
            Which duplicate to keep (default ``"first"``).

        Returns
        -------
        DataCleaner
            Self, for method chaining.
        """
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(self.df)
        if removed:
            logger.info(f"Dropped {removed} duplicate rows ({before} -> {len(self.df)})")
        else:
            logger.info("No duplicate rows found")
        self.raw_df = self.df.copy()
        return self

    def validate_schema(
        self,
        expected_schema: Optional[Dict[str, str]] = None,
        required_cols: Optional[List[str]] = None,
    ) -> List[str]:
        """Validate that the data meets expected column constraints.

        Parameters
        ----------
        expected_schema : dict of str -> str, optional
            Column name -> expected dtype (``"numeric"``, ``"datetime"``, ``"string"``).
        required_cols : list of str, optional
            Column names that must exist.

        Returns
        -------
        list of str
            List of validation issues (empty if all checks pass).
        """
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")
        issues: List[str] = []

        if required_cols:
            missing = [c for c in required_cols if c not in self.df.columns]
            if missing:
                issues.append(f"Missing required columns: {missing}")

        if expected_schema:
            for col, expected in expected_schema.items():
                if col not in self.df.columns:
                    issues.append(f"Column '{col}' not found (expected dtype: {expected})")
                    continue
                actual = self.df[col].dtype
                if expected == "numeric" and not pd.api.types.is_numeric_dtype(actual):
                    issues.append(f"Column '{col}' is {actual}, expected numeric")
                elif expected == "datetime" and not pd.api.types.is_datetime64_dtype(actual):
                    issues.append(f"Column '{col}' is {actual}, expected datetime")
                elif expected == "string" and not (
                    pd.api.types.is_string_dtype(actual) or pd.api.types.is_object_dtype(actual)
                ):
                    issues.append(f"Column '{col}' is {actual}, expected string")

        if issues:
            for issue in issues:
                logger.warning(issue)
            return issues
        logger.info("Schema validation passed")
        return []

    def auto_fix_dtypes(self) -> List[str]:
        """Auto-convert object columns to datetime or numeric where possible.

        For each object column, tries datetime parsing first (>70% parseable),
        then numeric conversion (>70% convertible, handles comma thousands).

        Returns
        -------
        list of str
            Descriptions of applied fixes (empty if none needed).
        """
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")
        fixes: List[str] = []

        for col in self.df.columns:
            if not pd.api.types.is_object_dtype(self.df[col]):
                continue
            sample = self.df[col].dropna().head(100)
            if sample.empty:
                continue

            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() / len(parsed) > 0.7:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                    fixes.append(f"{col}: object -> datetime")
                    continue
            except (ValueError, TypeError):
                pass

            try:
                cleaned = sample.astype(str).str.replace(",", "", regex=False)
                parsed = pd.to_numeric(cleaned, errors="coerce")
                if parsed.notna().sum() / len(parsed) > 0.7:
                    self.df[col] = pd.to_numeric(
                        self.df[col].astype(str).str.replace(",", "", regex=False),
                        errors="coerce",
                    )
                    fixes.append(f"{col}: object -> numeric")
                    continue
            except (ValueError, TypeError):
                pass

        if fixes:
            for f in fixes:
                logger.info(f"Auto-fix dtype: {f}")
        else:
            logger.info("No dtype fixes needed")
        self.raw_df = self.df.copy()
        return fixes

    def profile_report(self, filepath: Optional[str] = None) -> str:
        """Generate a self-contained HTML data profiling report.

        Includes:
        - Dataset overview cards (rows, cols, nulls, duplicates, memory)
        - Data quality warnings (high nulls, duplicates)
        - Column details table (type, nulls, quantiles, skew, outliers)
        - Numeric distribution histograms (requires matplotlib)
        - Correlation heatmap (requires matplotlib)
        - Categorical bar charts (requires matplotlib)

        Parameters
        ----------
        filepath : str, optional
            If provided, saves the HTML to disk.

        Returns
        -------
        str
            The complete HTML string.
        """
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")

        df = self.df
        n = len(df)
        numeric_df = df.select_dtypes(include="number")
        cat_df = df.select_dtypes(exclude="number")
        null_df = df.isnull().sum()
        null_pct = (null_df / n * 100).round(2)
        dup_count = int(df.duplicated().sum())
        total_memory = df.memory_usage(deep=True).sum() / 1024
        n_num = len(numeric_df.columns)
        n_cat = len(cat_df.columns)

        corr_html = ""
        hist_html = ""
        cat_bar_html = ""
        charts_html = ""

        try:
            import io
            import base64
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def _img_to_b64(fig: plt.Figure) -> str:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                plt.close(fig)
                return b64

            if n_num >= 2:
                corr = numeric_df.corr()
                fig, ax = plt.subplots(figsize=(max(6, n_num * 0.6), max(5, n_num * 0.5)))
                im = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(corr.columns, fontsize=8)
                for i in range(len(corr.columns)):
                    for j in range(len(corr.columns)):
                        val = corr.iloc[i, j]
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if abs(val) > 0.5 else "black")
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title("Correlation Heatmap", fontsize=12)
                b64 = _img_to_b64(fig)
                corr_html = (
                    '<div class="section"><h2>Correlation Heatmap</h2>'
                    f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"></div>'
                )

            num_plot_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() > 2]
            if num_plot_cols:
                n_plots = len(num_plot_cols)
                cols_per_row = 3
                n_rows = (n_plots + cols_per_row - 1) // cols_per_row
                fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(14, 3.5 * n_rows))
                axes = axes.flatten()
                for i, col in enumerate(num_plot_cols):
                    s = numeric_df[col].dropna()
                    axes[i].hist(s, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
                    axes[i].axvline(s.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={s.mean():.2f}")
                    axes[i].axvline(
                        s.median(), color="green", linestyle="--",
                        linewidth=1.5, label=f"med={s.median():.2f}",
                    )
                    axes[i].set_title(col, fontsize=10)
                    axes[i].tick_params(labelsize=7)
                    axes[i].legend(fontsize=6)
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)
                fig.suptitle("Numeric Column Distributions", fontsize=14, y=1.01)
                fig.tight_layout()
                b64 = _img_to_b64(fig)
                hist_html = (
                    '<div class="section"><h2>Distributions</h2>'
                    f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"></div>'
                )

            cat_plot_cols = [c for c in cat_df.columns if 1 < cat_df[c].nunique() <= 20]
            if cat_plot_cols:
                n_plots = len(cat_plot_cols)
                cols_per_row = 3
                n_rows = (n_plots + cols_per_row - 1) // cols_per_row
                fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(14, 3.5 * n_rows))
                axes = axes.flatten()
                for i, col in enumerate(cat_plot_cols):
                    vc = df[col].value_counts().head(10)
                    colors = plt.cm.Set3(range(len(vc)))
                    axes[i].barh(vc.index.astype(str), vc.values, color=colors)
                    axes[i].set_title(f"{col} (top 10)", fontsize=10)
                    axes[i].tick_params(labelsize=7)
                    for j, v in enumerate(vc.values):
                        axes[i].text(v + 0.3, j, str(v), va="center", fontsize=7)
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)
                fig.suptitle("Categorical Column Analysis", fontsize=14, y=1.01)
                fig.tight_layout()
                b64 = _img_to_b64(fig)
                cat_bar_html = (
                    '<div class="section"><h2>Categorical Columns</h2>'
                    f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"></div>'
                )

        except ImportError:
            charts_html = (
                '<div class="section"><div class="warn">Charts require matplotlib.'
                ' Install: pip install matplotlib seaborn</div></div>'
            )

        def _fmt(val: Any) -> str:
            return f"{val:.4f}" if isinstance(val, float) else str(val)

        rows_html = ""
        for col in df.columns:
            dtype = df[col].dtype
            null_c = int(null_df[col])
            null_p = null_pct[col]
            is_num = pd.api.types.is_numeric_dtype(dtype)
            bar_w = min(null_p, 100)

            if is_num and not df[col].dropna().empty:
                s = df[col].dropna()
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = int(((s < lo) | (s > hi)).sum())
                skew = float(s.skew())
                extra = (
                    f"<td>{_fmt(s.min())}</td>"
                    f"<td>{_fmt(q1)}</td>"
                    f"<td>{_fmt(s.median())}</td>"
                    f"<td>{_fmt(q3)}</td>"
                    f"<td>{_fmt(s.max())}</td>"
                    f"<td>{_fmt(s.mean())}</td>"
                    f"<td>{_fmt(s.std())}</td>"
                    f"<td>{skew:.2f}</td>"
                    f"<td>{outliers}</td>"
                )
            else:
                n_unique = df[col].nunique()
                top = df[col].mode().iloc[0] if not df[col].mode().empty else ""
                top_val = df[col].value_counts(normalize=True)
                top_pct = (top_val.iloc[0] * 100) if not top_val.empty else 0
                extra = (
                    f"<td colspan='9'><span class='chip'>{n_unique} unique</span>"
                    f" top: <b>{str(top)[:40]}</b> ({top_pct:.1f}%)</td>"
                )

            rows_html += (
                f"<tr>"
                f"<td><b>{col}</b></td>"
                f"<td>{dtype}</td>"
                f"<td>{null_c}</td>"
                f"<td>{null_p}%<div class='bar'><div class='fill' style='width:{bar_w}%'></div></div></td>"
                f"{extra}</tr>"
            )

        quality_issues: List[str] = []
        for col in df.columns:
            np_ = null_pct[col]
            if np_ > 20:
                quality_issues.append(
                    f"<li><span class='badge badge-err'>HIGH NULLS</span>"
                    f" <b>{col}</b>: {np_}% missing</li>"
                )
            elif np_ > 5:
                quality_issues.append(
                    f"<li><span class='badge badge-warn'>NULLS</span>"
                    f" <b>{col}</b>: {np_}% missing</li>"
                )
        if dup_count > 0:
            quality_issues.append(
                f"<li><span class='badge badge-warn'>DUPLICATES</span>"
                f" {dup_count} duplicate rows found</li>"
            )

        quality_html = (
            "<div class='section'><h2>Data Quality Warnings</h2><ul>" + "".join(quality_issues) + "</ul></div>"
            if quality_issues
            else "<div class='section'><h2>Data Quality</h2><p class='ok'>No major issues detected.</p></div>"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Profile Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    'Helvetica Neue', sans-serif; background: #f0f2f5; color: #1a1a2e; padding: 20px; }}
  .container {{ max-width: 1300px; margin: auto; }}
  .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #fff; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 28px; margin-bottom: 6px; }}
  .header p {{ opacity: 0.85; font-size: 14px; }}
  .section {{ background: #fff; border-radius: 10px; padding: 24px;
    margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
  .section h2 {{ font-size: 18px; color: #2c3e50; margin-bottom: 16px;
    padding-bottom: 8px; border-bottom: 2px solid #667eea; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 0; }}
  .stat-card {{ background: #f8f9fc; padding: 18px; border-radius: 10px;
    text-align: center; border: 1px solid #e8ecf4; }}
  .stat-card .num {{ font-size: 30px; font-weight: 700; color: #667eea; }}
  .stat-card .label {{ font-size: 12px; color: #7f8c8d; margin-top: 4px;
    text-transform: uppercase; letter-spacing: 0.5px; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f8f9fc; font-weight: 600; color: #555; padding: 10px 8px;
    text-align: left; border-bottom: 2px solid #e8ecf4;
    white-space: nowrap; position: sticky; top: 0; }}
  td {{ padding: 9px 8px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }}
  tr:hover {{ background: #f8f9ff; }}
  .bar {{ background: #eee; height: 14px; border-radius: 7px; overflow: hidden; min-width: 60px; }}
  .fill {{ height: 100%; background: linear-gradient(90deg, #e74c3c, #c0392b); border-radius: 7px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }}
  .badge-err {{ background: #fde8e8; color: #c0392b; }}
  .badge-warn {{ background: #fef5e7; color: #d68910; }}
  .badge-ok {{ background: #e8f8f5; color: #1abc9c; }}
  .chip {{ display: inline-block; background: #eef2ff; padding: 2px 10px;
    border-radius: 10px; font-size: 11px; color: #667eea; font-weight: 600; }}
  .ok {{ color: #27ae60; font-weight: 500; }}
  .warn {{ background: #fef5e7; color: #d68910; padding: 12px; border-radius: 6px; font-size: 13px; }}
  .footer {{ text-align: center; color: #aaa; font-size: 12px; padding: 20px; }}
  img {{ border-radius: 6px; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 8px; font-size: 13px; }}
  @media (max-width: 768px) {{ .header {{ padding: 20px; }} .section {{ padding: 16px; }} }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>Data Profile Report</h1>
  <p>{n} rows &times; {df.shape[1]} columns &nbsp;|&nbsp; {n_num} numeric,
    {n_cat} categorical &nbsp;|&nbsp; Generated on
    {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>

<div class="section">
  <h2>Dataset Overview</h2>
  <div class="stats">
    <div class="stat-card"><div class="num">{n:,}</div><div class="label">Rows</div></div>
    <div class="stat-card"><div class="num">{df.shape[1]}</div><div class="label">Columns</div></div>
    <div class="stat-card"><div class="num">{n_num}</div><div class="label">Numeric</div></div>
    <div class="stat-card"><div class="num">{n_cat}</div><div class="label">Categorical</div></div>
    <div class="stat-card"><div class="num">{int(null_df.sum()):,}</div><div class="label">Null Cells</div></div>
    <div class="stat-card"><div class="num">{dup_count:,}</div><div class="label">Duplicate Rows</div></div>
    <div class="stat-card"><div class="num">{total_memory:.1f}</div><div class="label">Memory (KB)</div></div>
    <div class="stat-card">
      <div class="num">{df.select_dtypes(include='datetime64').shape[1]}</div>
      <div class="label">Date Columns</div>
    </div>
  </div>
</div>

{quality_html}

<div class="section">
  <h2>Column Details</h2>
  <div class="table-wrap">
  <table>
  <thead><tr>
    <th>Column</th><th>Type</th><th>Nulls</th><th>Null %</th>
    <th>Min</th><th>Q1</th><th>Median</th><th>Q3</th><th>Max</th><th>Mean</th><th>Std</th><th>Skew</th><th>Outliers</th>
  </tr></thead>
  <tbody>
  {rows_html}
  </tbody>
  </table>
  </div>
</div>

{hist_html}
{corr_html}
{cat_bar_html}
{charts_html}

<div class="footer">Generated by <b>clean_data_ml</b> v1.2.0</div>
</div>
</body>
</html>"""

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"Profile report saved to {filepath}")
        return html
