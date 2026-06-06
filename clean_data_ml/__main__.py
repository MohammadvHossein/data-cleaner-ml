"""Entry point for ``python -m clean_data_ml``."""

from . import __version__


def main() -> None:
    """Print version and usage information."""
    print(f"clean_data_ml v{__version__}")
    print("Automated data cleaning and standardization for ML pipelines.")
    print()
    print("Usage:")
    print("  from clean_data_ml import DataCleaner")
    print('  dc = DataCleaner()')
    print('  dc.load("data.csv").set_target("target").prepare()')


if __name__ == "__main__":
    main()
