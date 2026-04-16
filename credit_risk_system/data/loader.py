"""Load and filter raw LendingClub loan data."""

import logging
from pathlib import Path

import pandas as pd

from config.feature_config import POSITIVE_LABELS, RAW_COLUMNS, TARGET_COLUMN

logger = logging.getLogger(__name__)


class LendingClubLoader:
    """
    Ingests LendingClub CSV files, filters to closed loans, and binarises the target.

    Usage:
        loader = LendingClubLoader("data/raw/loan.csv")
        df = loader.load()
    """

    def __init__(self, filepath: str, chunksize: int = 50_000):
        self.filepath = Path(filepath)
        self.chunksize = chunksize

    def load(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.filepath}. "
                "Download the LendingClub dataset from Kaggle and place it in data/raw/."
            )
        logger.info("Loading %s in chunks of %d …", self.filepath, self.chunksize)

        chunks = []
        for chunk in pd.read_csv(
            self.filepath,
            chunksize=self.chunksize,
            low_memory=False,
            on_bad_lines="skip",
        ):
            # Keep only columns we care about (silently skip missing ones)
            available = [c for c in RAW_COLUMNS if c in chunk.columns]
            chunks.append(chunk[available])

        df = pd.concat(chunks, ignore_index=True)
        logger.info("Loaded %d rows before filtering.", len(df))

        df = self.filter_closed_loans(df)
        df = self.encode_target(df)
        logger.info("After filtering: %d rows, target distribution:\n%s", len(df), df[TARGET_COLUMN].value_counts())
        return df

    def filter_closed_loans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only loans that have a definitive outcome (paid off or defaulted)."""
        if TARGET_COLUMN not in df.columns:
            return df
        closed_labels = POSITIVE_LABELS | {"Fully Paid"}
        return df[df[TARGET_COLUMN].isin(closed_labels)].copy()

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert loan_status to binary: default/charged-off = 1, fully paid = 0."""
        if TARGET_COLUMN not in df.columns:
            return df
        df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
            lambda s: 1 if s in POSITIVE_LABELS else 0
        )
        return df
