from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="clean_data_ml",
    version="1.2.0",
    description="Automatic data cleaning and standardization for ML pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mohammad Hossein Habibpour",
    author_email="habibpour.programming@gmail.com",
    url="https://github.com/MohammadvHossein/clean-data-ml",
    project_urls={
        "Homepage": "https://github.com/MohammadvHossein/clean-data-ml",
        "Repository": "https://github.com/MohammadvHossein/clean-data-ml",
        "Bug Tracker": "https://github.com/MohammadvHossein/clean-data-ml/issues",
    },
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Typing :: Typed",
    ],
)
