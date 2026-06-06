import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib/seaborn not installed. Install with: pip install data-cleaner[plot]")


def plot_null_report(dc):
    if not _HAS_MPL:
        return
    if dc.df is None:
        logger.error("No data loaded")
        return
    null_counts = dc.df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        logger.info("No null values found")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    null_counts.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Null Counts by Column")
    axes[0].set_ylabel("Count")
    (null_counts / len(dc.df) * 100).plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("Null Percentage by Column")
    axes[1].set_ylabel("Percentage (%)")
    plt.tight_layout()
    plt.show()


def plot_distributions(dc, cols=None):
    if not _HAS_MPL:
        return
    if dc.df is None:
        logger.error("No data loaded")
        return
    df = dc.df.select_dtypes(include="number")
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    if df.empty:
        logger.info("No numeric columns to plot")
        return
    n = len(df.columns)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(df.columns):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i][0])
        axes[i][0].set_title(f"{col} - Distribution")
        sns.boxplot(x=df[col].dropna(), ax=axes[i][1])
        axes[i][1].set_title(f"{col} - Boxplot")
    plt.tight_layout()
    plt.show()


def plot_correlation(dc, figsize=(10, 8)):
    if not _HAS_MPL:
        return
    if dc.df is None:
        logger.error("No data loaded")
        return
    df = dc.df.select_dtypes(include="number")
    if df.empty or df.shape[1] < 2:
        logger.info("Need at least 2 numeric columns for correlation")
        return
    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_before_after(dc):
    if not _HAS_MPL:
        return
    if not dc.is_fitted:
        logger.error("Must call .prepare() first")
        return
    raw = dc.raw_df.select_dtypes(include="number")
    clean = dc.X_clean.select_dtypes(include="number")
    common = [c for c in clean.columns if c in raw.columns]
    if not common:
        logger.info("No common numeric columns to compare")
        return
    n = len(common)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(common):
        axes[i][0].hist(raw[col].dropna(), bins=20, alpha=0.7)
        axes[i][0].set_title(f"{col} - Before")
        axes[i][1].hist(clean[col].dropna(), bins=20, alpha=0.7, color="green")
        axes[i][1].set_title(f"{col} - After")
    plt.tight_layout()
    plt.show()
