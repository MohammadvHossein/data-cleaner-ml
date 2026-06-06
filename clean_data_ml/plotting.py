"""Optional visualization module for DataCleaner.

Requires matplotlib and seaborn (install with ``pip install clean-data-ml[plot]``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .cleaner import DataCleaner

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

    def _noop(*args, **kwargs) -> None:  # type: ignore[misc]
        return None


def plot_null_report(dc: "DataCleaner") -> None:
    """Plot bar charts of null counts and null percentages for each column.

    Parameters
    ----------
    dc : DataCleaner
        A DataCleaner instance with loaded data.
    """
    if not _HAS_MPL:
        logger.warning("matplotlib/seaborn not installed. Install with: pip install clean-data-ml[plot]")
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


def plot_distributions(dc: "DataCleaner", cols: Optional[List[str]] = None) -> None:
    """Plot histograms with KDE and boxplots for numeric columns.

    Parameters
    ----------
    dc : DataCleaner
        A DataCleaner instance with loaded data.
    cols : list of str, optional
        Subset of numeric columns to plot. If None, all numeric columns are used.
    """
    if not _HAS_MPL:
        logger.warning("matplotlib/seaborn not installed. Install with: pip install clean-data-ml[plot]")
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
    axes_flat = axes.flatten() if n > 1 else axes
    for i, col in enumerate(df.columns):
        ax_hist: plt.Axes = axes_flat[i * 2] if n > 1 else axes[0]  # type: ignore[assignment]
        ax_box: plt.Axes = axes_flat[i * 2 + 1] if n > 1 else axes[1]  # type: ignore[assignment]
        sns.histplot(df[col].dropna(), kde=True, ax=ax_hist)
        ax_hist.set_title(f"{col} - Distribution")
        sns.boxplot(x=df[col].dropna(), ax=ax_box)
        ax_box.set_title(f"{col} - Boxplot")
    plt.tight_layout()
    plt.show()


def plot_correlation(dc: "DataCleaner", figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot a correlation heatmap for all numeric columns.

    Parameters
    ----------
    dc : DataCleaner
        A DataCleaner instance with loaded data.
    figsize : tuple of int, optional
        Figure size passed to matplotlib (default (10, 8)).
    """
    if not _HAS_MPL:
        logger.warning("matplotlib/seaborn not installed. Install with: pip install clean-data-ml[plot]")
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


def plot_before_after(dc: "DataCleaner") -> None:
    """Compare raw vs cleaned distributions for common numeric columns.

    Requires ``.prepare()`` to have been called first.

    Parameters
    ----------
    dc : DataCleaner
        A fitted DataCleaner instance.
    """
    if not _HAS_MPL:
        logger.warning("matplotlib/seaborn not installed. Install with: pip install clean-data-ml[plot]")
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
    axes_flat = axes.flatten() if n > 1 else axes
    for i, col in enumerate(common):
        ax_before: plt.Axes = axes_flat[i * 2] if n > 1 else axes[0]  # type: ignore[assignment]
        ax_after: plt.Axes = axes_flat[i * 2 + 1] if n > 1 else axes[1]  # type: ignore[assignment]
        ax_before.hist(raw[col].dropna(), bins=20, alpha=0.7)
        ax_before.set_title(f"{col} - Before")
        ax_after.hist(clean[col].dropna(), bins=20, alpha=0.7, color="green")
        ax_after.set_title(f"{col} - After")
    plt.tight_layout()
    plt.show()
