"""Download and split the California housing dataset."""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

LOGGER = logging.getLogger(__name__)
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"


def fetch_housing_data(output_dir: str | Path, url: str = HOUSING_URL) -> Path:
    """Download and extract the housing dataset.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory receiving the archive and extracted CSV.
    url : str
        Dataset archive URL.

    Returns
    -------
    pathlib.Path
        Path to the extracted CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "housing.tgz"
    csv_path = output_dir / "housing.csv"
    if not csv_path.exists():
        LOGGER.info("Downloading housing data from %s", url)
        urlretrieve(url, archive_path)
        with tarfile.open(archive_path) as archive:
            archive.extractall(path=output_dir)
    return csv_path


def create_datasets(raw_path: str | Path, processed_dir: str | Path) -> tuple[Path, Path]:
    """Create stratified training and validation CSV files.

    Parameters
    ----------
    raw_path : str or pathlib.Path
        Path to the raw housing CSV.
    processed_dir : str or pathlib.Path
        Output directory for training and validation data.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Paths to the training and validation CSV files.
    """
    raw_path = Path(raw_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    housing = pd.read_csv(raw_path)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, float("inf")],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, validation_indices = next(split.split(housing, housing["income_cat"]))
    train = housing.iloc[train_indices].drop(columns="income_cat")
    validation = housing.iloc[validation_indices].drop(columns="income_cat")
    train_path = processed_dir / "train.csv"
    validation_path = processed_dir / "validation.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    LOGGER.info("Wrote %s and %s", train_path, validation_path)
    return train_path, validation_path


def ingest_data(raw_dir: str | Path, processed_dir: str | Path) -> tuple[Path, Path]:
    """Download raw data and create processed training and validation datasets."""
    raw_path = fetch_housing_data(raw_dir)
    return create_datasets(raw_path, processed_dir)
