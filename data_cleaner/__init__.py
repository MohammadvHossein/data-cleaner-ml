"""data_cleaner: automated data cleaning and standardization for ML pipelines."""

from .cleaner import DataCleaner
from . import stats

__all__ = ["DataCleaner", "stats"]
__version__ = "1.1.0"
