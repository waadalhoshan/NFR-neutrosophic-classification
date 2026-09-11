"""
data.py — Loading utilities for the NICE non-functional requirements dataset.

The NICE dataset (Rejithkumar & Anish, 2025) is a multi-label re-annotation
of the augmented PROMISE NFR corpus. This module loads the pre-filtered
NFR-only subset (functional requirements and the empty "Other" category
already removed) used throughout this study.

Source dataset: https://zenodo.org/records/14590935
"""
import pandas as pd
import numpy as np

LABEL_COLS = [
    'Availability (A)', 'Fault Tolerance (FT)', 'Legal (L)', 'Look & Feel (LF)',
    'Maintainability (MN)', 'Operability (O)', 'Performance (PE)', 'Portability (PO)',
    'Scalability (SC)', 'Security (SE)', 'Usability (US)'
]

DEFAULT_DATA_PATH = "data/nice_nfr.csv"


def load_nice_dataset(path: str = DEFAULT_DATA_PATH):
    """
    Load the cleaned NICE NFR dataset.

    Returns
    -------
    df : pandas.DataFrame
        Full dataframe including requirement text and label columns.
    texts : np.ndarray of str
        Requirement text, one entry per row.
    Y : np.ndarray of shape (n_samples, n_categories)
        Binary multi-label ground-truth matrix.
    """
    df = pd.read_csv(path)
    texts = df['RequirementText'].astype(str).values
    Y = df[LABEL_COLS].values
    return df, texts, Y


def label_cardinality_summary(Y: np.ndarray) -> dict:
    """Return counts of requirements by number of true labels (1, 2, 3, ...)."""
    num_labels = Y.sum(axis=1)
    unique, counts = np.unique(num_labels, return_counts=True)
    return dict(zip(unique.astype(int).tolist(), counts.tolist()))


if __name__ == "__main__":
    df, texts, Y = load_nice_dataset()
    print(f"Loaded {len(texts)} requirements across {Y.shape[1]} categories.")
    print("Label cardinality distribution:", label_cardinality_summary(Y))
