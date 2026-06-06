from setuptools import setup, find_packages

setup(
    name="data_cleaner",
    version="1.1.0",
    description="Automatic data cleaning and standardization for ML pipelines",
    author="Mohammad Hossein Habibpour",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "openpyxl>=3.0.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "plot": ["matplotlib>=3.5.0", "seaborn>=0.11.0"],
        "imbalance": ["imbalanced-learn>=0.10.0"],
        "all": ["matplotlib>=3.5.0", "seaborn>=0.11.0", "imbalanced-learn>=0.10.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
