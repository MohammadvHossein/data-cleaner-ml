"""Entry point for ``python -m data_cleaner``."""

from . import __version__

if __name__ == "__main__":
    print(f"data_cleaner v{__version__}")
    print("Automated data cleaning and standardization for ML pipelines.")
    print()
    print("Usage:")
    print("  from data_cleaner import DataCleaner")
    print("  dc = DataCleaner()")
    print('  dc.load("data.csv").set_target("target").prepare()')
