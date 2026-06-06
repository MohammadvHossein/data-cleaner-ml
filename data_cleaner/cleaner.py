import pandas as pd
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .auto_scaler import select_best_scaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


class CleanPipeline:
    def __init__(self):
        self.columns_to_drop = []
        self.target_col = None
        self.feature_cols = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.scalers = {}
        self.label_encoders = {}
        self.onehot_encoder = None
        self.onehot_cols = []
        self.imputer = None
        self.impute_cols = []
        self.impute_strategy = {}
        self.used_knn = False
        self.problem_type = None
        self.dropped_useless_cols = []
        self.outlier_bounds = {}
        self.poly_features = None

    def transform(self, df):
        df = df.copy()

        for col in self.columns_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        if self.imputer is not None:
            present_impute = [c for c in self.impute_cols if c in df.columns]
            if present_impute:
                imputed = self.imputer.transform(df[present_impute])
                df[present_impute] = imputed

        for col in self.binary_cols:
            if col in df.columns:
                mapping = self.label_encoders.get(col, {})
                df[col] = df[col].fillna(list(mapping.keys())[0] if mapping else "unknown")
                df[col] = df[col].map(mapping).fillna(0)

        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")
                df[col] = df[col].astype(str)

        if self.onehot_encoder is not None and self.categorical_cols:
            present_cat = [c for c in self.categorical_cols if c in df.columns]
            if present_cat:
                encoded = self.onehot_encoder.transform(df[present_cat])
                encoded_df = pd.DataFrame(
                    encoded, columns=self.onehot_cols, index=df.index
                )
                df = pd.concat([df.drop(columns=present_cat), encoded_df], axis=1)

        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])

        missing = set(self.feature_cols or []) - set(df.columns)
        if missing:
            for col in missing:
                df[col] = 0

        return df[self.feature_cols] if self.feature_cols else df

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class DataCleaner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.df = None
        self.raw_df = None
        self.target_col = None
        self.columns_to_drop = []
        self.pipeline = CleanPipeline()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_clean = None
        self.y_clean = None
        self.is_fitted = False

    def load(self, filepath):
        if filepath.endswith(".csv"):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith((".xls", ".xlsx")):
            self.df = pd.read_excel(filepath)
        else:
            raise ValueError("Only CSV and Excel files are supported")
        self.raw_df = self.df.copy()
        return self

    def load_df(self, dataframe):
        self.df = dataframe.copy()
        self.raw_df = self.df.copy()
        return self

    def set_target(self, column):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in data")
        self.target_col = column
        return self

    def drop_columns(self, columns):
        for col in columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in data")
        self.columns_to_drop = columns
        return self

    def _auto_detect_types(self, X=None):
        if X is not None:
            df = X
        else:
            df = self.df.copy()
            for col in self.columns_to_drop:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            if self.target_col and self.target_col in df.columns:
                df.drop(columns=[self.target_col], inplace=True)

        numeric_cols = []
        categorical_cols = []
        binary_cols = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].nunique()
                if unique_vals == 2:
                    binary_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                unique_vals = df[col].nunique()
                if unique_vals == 2:
                    binary_cols.append(col)
                else:
                    categorical_cols.append(col)

        return numeric_cols, categorical_cols, binary_cols

    def _select_scalers(self, numeric_cols, df, n_jobs=1):
        if n_jobs != 1 and _HAS_JOBLIB and len(numeric_cols) > 5:
            results = Parallel(n_jobs=n_jobs)(
                delayed(select_best_scaler)(df[col]) for col in numeric_cols
            )
            return dict(zip(numeric_cols, results))
        scalers = {}
        for col in numeric_cols:
            scalers[col] = select_best_scaler(df[col])
        return scalers

    @staticmethod
    def _calc_dynamic_threshold(n_rows):
        threshold = 0.05 * (1000 / max(n_rows, 1))
        return round(max(0.01, min(0.25, threshold)), 4)

    def _handle_nulls(self, df, max_drop_ratio=None):
        df = df.copy()
        if max_drop_ratio is None:
            max_drop_ratio = self._calc_dynamic_threshold(len(df))

        null_info = {}
        impute_cols = []
        drop_cols = []

        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count == 0:
                continue
            null_ratio = null_count / len(df)
            null_info[col] = {"count": null_count, "ratio": null_ratio}

            if null_ratio <= max_drop_ratio:
                drop_cols.append(col)
            else:
                impute_cols.append(col)

        if drop_cols:
            df = df.dropna(subset=drop_cols)

        return df, null_info, impute_cols

    def _impute_nulls(self, df, impute_cols):
        if not impute_cols:
            return df, None

        numeric_impute = [c for c in impute_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_impute = [c for c in impute_cols if not pd.api.types.is_numeric_dtype(df[c])]

        if numeric_impute:
            knn = KNNImputer(n_neighbors=5)
            df[numeric_impute] = knn.fit_transform(df[numeric_impute])
            self.pipeline.imputer = knn
            self.pipeline.impute_cols = numeric_impute
            self.pipeline.used_knn = True

        for col in categorical_impute:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown", inplace=True)

        return df

    @staticmethod
    def _detect_problem_type(y):
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        unique = y.nunique()
        if unique > 25:
            return "regression"
        if unique < 10:
            return "classification"
        ratios = y.value_counts(normalize=True)
        if ratios.max() > 0.5:
            return "classification"
        return "regression"

    def _auto_drop_useless(self, X):
        drop_cols = set()
        n = len(X)

        for col in X.columns:
            if X[col].nunique() <= 1:
                drop_cols.add(col)

        for col in X.columns:
            if col in drop_cols:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                if X[col].nunique() / n > 0.9 and n > 10:
                    drop_cols.add(col)

        X = X.drop(columns=list(drop_cols))
        self.pipeline.dropped_useless_cols = list(drop_cols)
        return X

    def _handle_outliers(self, X, method="clip", n_jobs=1):
        if n_jobs != 1 and _HAS_JOBLIB:
            cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 2]
            if not cols:
                return X
            def _compute_bounds(col):
                Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                IQR = Q3 - Q1
                return col, Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            results = Parallel(n_jobs=n_jobs)(delayed(_compute_bounds)(c) for c in cols)
            bounds = {col: (lo, hi) for col, lo, hi in results}
            self.pipeline.outlier_bounds = {col: {"lower": lo, "upper": hi} for col, (lo, hi) in bounds.items()}
            for col, (lo, hi) in bounds.items():
                if method == "clip":
                    X[col] = X[col].clip(lo, hi)
                elif method == "remove":
                    X = X[(X[col] >= lo) & (X[col] <= hi)]
            return X
        bounds = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            if X[col].nunique() <= 2:
                continue
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            bounds[col] = {"lower": lower, "upper": upper}
            if method == "clip":
                X[col] = X[col].clip(lower, upper)
            elif method == "remove":
                before = len(X)
                X = X[(X[col] >= lower) & (X[col] <= upper)]
        self.pipeline.outlier_bounds = bounds
        return X

    def _feature_engineering(self, X):
        from sklearn.preprocessing import PolynomialFeatures

        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 2]
        if not numeric_cols or len(numeric_cols) < 2:
            return X

        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        poly_features = poly.fit_transform(X[numeric_cols])
        poly_names = poly.get_feature_names_out(numeric_cols).tolist()
        existing = set(X.columns)
        new_names = [n for n in poly_names if n not in existing]

        poly_df = pd.DataFrame(
            poly_features[:, len(numeric_cols):],
            columns=new_names,
            index=X.index
        )
        self.pipeline.poly_features = poly
        X = pd.concat([X, poly_df], axis=1)
        return X

    def _apply_smote(self, X_train, y_train):
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE applied. New train size: {len(X_train)}")
        except ImportError:
            logger.warning("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
        return X_train, y_train

    def _encode_binary(self, df, binary_cols):
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.pipeline.label_encoders[col] = dict(
                zip(le.classes_, le.transform(le.classes_))
            )
        return df

    def _encode_categorical(self, df, categorical_cols):
        if not categorical_cols:
            return df

        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")
            df[col] = df[col].astype(str)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(df[categorical_cols])
        self.pipeline.onehot_encoder = ohe
        self.pipeline.onehot_cols = ohe.get_feature_names_out(categorical_cols).tolist()

        encoded_df = pd.DataFrame(
            encoded, columns=self.pipeline.onehot_cols, index=df.index
        )
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
        return df

    def _scale_features(self, df, scalers):
        for col, scaler in scalers.items():
            df[col] = scaler.fit_transform(df[[col]])
        return df

    def prepare(self, test_size=0.2, val_size=None, handle_nulls=True, auto_scale=True,
                auto_encode=True, null_drop_ratio=None, auto_drop_useless=True,
                handle_outliers=None, feature_engineering=False, handle_imbalance=False,
                n_jobs=1):
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

        if auto_drop_useless:
            X = self._auto_drop_useless(X)

        numeric_cols, categorical_cols, binary_cols = self._auto_detect_types(X)

        self.pipeline.numeric_cols = numeric_cols
        self.pipeline.categorical_cols = categorical_cols
        self.pipeline.binary_cols = binary_cols

        if handle_nulls:
            y_null = y.isnull().sum()
            if y_null > 0:
                y_null_ratio = y_null / len(y)
                if y_null_ratio <= 0.05:
                    valid_idx = y.dropna().index
                    X = X.loc[valid_idx]
                    y = y.loc[valid_idx]
                else:
                    raise ValueError(f"Target column '{self.target_col}' has {y_null_ratio:.1%} null values. Please clean it manually.")

            merged = pd.concat([X, y], axis=1)
            merged, null_info, impute_cols = self._handle_nulls(merged, null_drop_ratio)
            X = merged.drop(columns=[self.target_col])
            y = merged[self.target_col]
            if impute_cols:
                X = self._impute_nulls(X, impute_cols)

        if handle_outliers in ("clip", "remove"):
            X = self._handle_outliers(X, method=handle_outliers, n_jobs=n_jobs)

        if feature_engineering:
            X = self._feature_engineering(X)

        if auto_encode:
            X = self._encode_binary(X, binary_cols)
            X = self._encode_categorical(X, categorical_cols)
        else:
            for col in binary_cols + categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        if auto_scale:
            scalers = self._select_scalers(numeric_cols, X)
            self.pipeline.scalers = scalers
            X = self._scale_features(X, scalers)

        self.X_clean = X
        self.y_clean = y
        self.pipeline.feature_cols = X.columns.tolist()

        def split_and_return(X, y):
            if val_size is not None:
                test_val_size = test_size + val_size
                X_train, X_test_val, y_train, y_test_val = train_test_split(
                    X, y, test_size=test_val_size, random_state=self.random_state
                )
                val_ratio = val_size / test_val_size if test_val_size > 0 else 0
                X_val, X_test, y_val, y_test = train_test_split(
                    X_test_val, y_test_val, test_size=1 - val_ratio,
                    random_state=self.random_state
                )
                self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
                self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
                self.is_fitted = True
                return X_train, X_val, X_test, y_train, y_val, y_test
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.is_fitted = True
                return X_train, X_test, y_train, y_test

        if handle_imbalance and self.pipeline.problem_type == "classification":
            result = split_and_return(X, y)
            if val_size is not None:
                X_train, X_val, X_test, y_train, y_val, y_test = result
            else:
                X_train, X_test, y_train, y_test = result
            X_train, y_train = self._apply_smote(X_train, y_train)
            self.X_train, self.y_train = X_train, y_train
            if val_size is not None:
                return X_train, X_val, X_test, y_train, y_val, y_test
            return X_train, X_test, y_train, y_test

        return split_and_return(X, y)

    def get_pipeline(self):
        if not self.is_fitted:
            raise RuntimeError("Must call .prepare() first")
        return self.pipeline

    def save_pipeline(self, path):
        self.get_pipeline().save(path)

    @staticmethod
    def load_pipeline(path):
        return CleanPipeline.load(path)

    def export_cleaned(self, filepath, include_target=False):
        if not self.is_fitted:
            raise RuntimeError("Must call .prepare() first")
        df = self.X_clean.copy()
        if include_target and self.y_clean is not None:
            df[self.target_col] = self.y_clean.values
        if filepath.endswith(".csv"):
            df.to_csv(filepath, index=False)
        elif filepath.endswith((".xls", ".xlsx")):
            df.to_excel(filepath, index=False)
        else:
            raise ValueError("Only CSV and Excel formats are supported")
        logger.info(f"Cleaned data exported to {filepath}")

    def summary(self):
        if self.df is None:
            return "No data loaded"
        info = {}
        info["shape"] = self.df.shape
        info["columns"] = list(self.df.columns)
        info["dtypes"] = {str(k): str(v) for k, v in self.df.dtypes.items()}
        info["null_counts"] = self.df.isnull().sum().to_dict()
        info["null_percent"] = (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        return info

    def drop_duplicates(self, subset=None, keep="first"):
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after = len(self.df)
        removed = before - after
        if removed:
            logger.info(f"Dropped {removed} duplicate rows ({before} -> {after})")
        else:
            logger.info("No duplicate rows found")
        self.raw_df = self.df.copy()
        return self

    def validate_schema(self, expected_schema=None, required_cols=None):
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")
        issues = []
        if required_cols:
            missing = [c for c in required_cols if c not in self.df.columns]
            if missing:
                issues.append(f"Missing required columns: {missing}")
        if expected_schema:
            for col, expected_dtype in expected_schema.items():
                if col not in self.df.columns:
                    issues.append(f"Column '{col}' not found (expected dtype: {expected_dtype})")
                    continue
                actual = self.df[col].dtype
                if expected_dtype == "numeric" and not pd.api.types.is_numeric_dtype(actual):
                    issues.append(f"Column '{col}' is {actual}, expected numeric")
                elif expected_dtype == "datetime" and not pd.api.types.is_datetime64_dtype(actual):
                    issues.append(f"Column '{col}' is {actual}, expected datetime")
                elif expected_dtype == "string" and not pd.api.types.is_string_dtype(actual) and not pd.api.types.is_object_dtype(actual):
                    issues.append(f"Column '{col}' is {actual}, expected string")
        if issues:
            for issue in issues:
                logger.warning(issue)
            return issues
        logger.info("Schema validation passed")
        return []

    def auto_fix_dtypes(self):
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")
        fixes = []
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
                sample_str = sample.astype(str).str.replace(",", "", regex=False)
                parsed_num = pd.to_numeric(sample_str, errors="coerce")
                if parsed_num.notna().sum() / len(parsed_num) > 0.7:
                    self.df[col] = pd.to_numeric(
                        self.df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
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

    def profile_report(self, filepath=None):
        if self.df is None:
            raise ValueError("No data loaded. Use .load() or .load_df() first.")
        df = self.df
        n = len(df)
        numeric_df = df.select_dtypes(include="number")
        cat_df = df.select_dtypes(exclude="number")
        null_df = df.isnull().sum()
        null_pct = (null_df / n * 100).round(2)
        dup_count = df.duplicated().sum()
        total_memory = df.memory_usage(deep=True).sum() / 1024
        n_num = len(numeric_df.columns)
        n_cat = len(cat_df.columns)

        charts_html = ""
        corr_html = ""
        hist_html = ""
        cat_bar_html = ""

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io
            import base64

            def _img_to_b64(fig):
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
                        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title("Correlation Heatmap", fontsize=12)
                b64 = _img_to_b64(fig)
                corr_html = f'<div class="section"><h2>Correlation Heatmap</h2><img src="data:image/png;base64,{b64}" style="max-width:100%;"></div>'

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
                    axes[i].axvline(s.median(), color="green", linestyle="--", linewidth=1.5, label=f"med={s.median():.2f}")
                    axes[i].set_title(col, fontsize=10)
                    axes[i].tick_params(labelsize=7)
                    axes[i].legend(fontsize=6)
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)
                fig.suptitle("Numeric Column Distributions", fontsize=14, y=1.01)
                fig.tight_layout()
                b64 = _img_to_b64(fig)
                hist_html = f'<div class="section"><h2>Distributions</h2><img src="data:image/png;base64,{b64}" style="max-width:100%;"></div>'

            cat_plot_cols = [c for c in cat_df.columns if cat_df[c].nunique() <= 20 and cat_df[c].nunique() > 1]
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
                cat_bar_html = f'<div class="section"><h2>Categorical Columns</h2><img src="data:image/png;base64,{b64}" style="max-width:100%;"></div>'

        except ImportError:
            charts_html = '<div class="section"><div class="warn">Charts require matplotlib. Install: pip install matplotlib seaborn</div></div>'

        def _fmt(val):
            if isinstance(val, float):
                return f"{val:.4f}"
            return str(val)

        rows_html = ""
        for col in df.columns:
            dtype = df[col].dtype
            null_c = int(null_df[col])
            null_p = null_pct[col]
            is_num = pd.api.types.is_numeric_dtype(dtype)
            bar_w = min(null_p, 100)

            if is_num and not df[col].dropna().empty:
                s = df[col].dropna()
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outliers = ((s < lo) | (s > hi)).sum()
                skew = s.skew()
                is_skewed = abs(skew) > 1
                extra = (
                    f"<td>{_fmt(s.min())}</td>"
                    f"<td>{_fmt(Q1)}</td>"
                    f"<td>{_fmt(s.median())}</td>"
                    f"<td>{_fmt(Q3)}</td>"
                    f"<td>{_fmt(s.max())}</td>"
                    f"<td>{_fmt(s.mean())}</td>"
                    f"<td>{_fmt(s.std())}</td>"
                    f"<td>{skew:.2f}</td>"
                    f"<td>{outliers}</td>"
                )
            else:
                unique = df[col].nunique()
                top_val = df[col].mode().iloc[0] if not df[col].mode().empty else ""
                top_pct = (df[col].value_counts(normalize=True).iloc[0] * 100) if not df[col].value_counts().empty else 0
                extra = f"<td colspan='9'><span class='chip'>{unique} unique</span> top: <b>{str(top_val)[:40]}</b> ({top_pct:.1f}%)</td>"

            rows_html += (
                f"<tr>"
                f"<td><b>{col}</b></td>"
                f"<td>{dtype}</td>"
                f"<td>{null_c}</td>"
                f"<td>{null_p}%<div class='bar'><div class='fill' style='width:{bar_w}%'></div></div></td>"
                f"{extra}</tr>"
            )

        quality_issues = []
        for col in df.columns:
            null_c = int(null_df[col])
            null_p = null_pct[col]
            if null_p > 20:
                quality_issues.append(f"<li><span class='badge badge-err'>HIGH NULLS</span> <b>{col}</b>: {null_p}% missing</li>")
            elif null_p > 5:
                quality_issues.append(f"<li><span class='badge badge-warn'>NULLS</span> <b>{col}</b>: {null_p}% missing</li>")
        if df.duplicated().sum() > 0:
            quality_issues.append(f"<li><span class='badge badge-warn'>DUPLICATES</span> {df.duplicated().sum()} duplicate rows found</li>")
        if quality_issues:
            quality_html = "<div class='section'><h2>Data Quality Warnings</h2><ul>" + "".join(quality_issues) + "</ul></div>"
        else:
            quality_html = "<div class='section'><h2>Data Quality</h2><p class='ok'>No major issues detected.</p></div>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Profile Report</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif; background: #f0f2f5; color: #1a1a2e; padding: 20px; }}
  .container {{ max-width: 1300px; margin: auto; }}
  .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ font-size: 28px; margin-bottom: 6px; }}
  .header p {{ opacity: 0.85; font-size: 14px; }}
  .section {{ background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
  .section h2 {{ font-size: 18px; color: #2c3e50; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid #667eea; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin-bottom: 0; }}
  .stat-card {{ background: #f8f9fc; padding: 18px; border-radius: 10px; text-align: center; border: 1px solid #e8ecf4; }}
  .stat-card .num {{ font-size: 30px; font-weight: 700; color: #667eea; }}
  .stat-card .label {{ font-size: 12px; color: #7f8c8d; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f8f9fc; font-weight: 600; color: #555; padding: 10px 8px; text-align: left; border-bottom: 2px solid #e8ecf4; white-space: nowrap; position: sticky; top: 0; }}
  td {{ padding: 9px 8px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }}
  tr:hover {{ background: #f8f9ff; }}
  .bar {{ background: #eee; height: 14px; border-radius: 7px; overflow: hidden; min-width: 60px; }}
  .fill {{ height: 100%; background: linear-gradient(90deg, #e74c3c, #c0392b); border-radius: 7px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }}
  .badge-err {{ background: #fde8e8; color: #c0392b; }}
  .badge-warn {{ background: #fef5e7; color: #d68910; }}
  .badge-ok {{ background: #e8f8f5; color: #1abc9c; }}
  .chip {{ display: inline-block; background: #eef2ff; padding: 2px 10px; border-radius: 10px; font-size: 11px; color: #667eea; font-weight: 600; }}
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
  <p>{n} rows &times; {df.shape[1]} columns &nbsp;|&nbsp; {n_num} numeric, {n_cat} categorical &nbsp;|&nbsp; Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
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
    <div class="stat-card"><div class="num">{df.select_dtypes(include='datetime64').shape[1]}</div><div class="label">Date Columns</div></div>
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

<div class="footer">Generated by <b>data_cleaner</b> v1.1.0</div>
</div>
</body>
</html>"""
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"Profile report saved to {filepath}")
        return html
