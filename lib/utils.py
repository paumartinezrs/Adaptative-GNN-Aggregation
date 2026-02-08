"""
Common utilities: seeding, CSV helpers.
"""

import os
import csv
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_row_to_csv(filepath, row_dict):
    """
    Append a single row (dict) to a CSV file.

    * Creates the file with headers if it does not exist.
    * If the file already exists, only columns present in the existing
      header are written (extra keys are silently ignored).
    """
    filepath = str(filepath)
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0

    if file_exists:
        with open(filepath, "r", newline="") as f:
            fieldnames = csv.DictReader(f).fieldnames
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writerow(row_dict)
            f.flush()
    else:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_dict.keys())
            writer.writeheader()
            writer.writerow(row_dict)
            f.flush()
