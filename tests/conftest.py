"""Shared fixtures for the test suite."""

import sys
import os
import io
import pytest
import pandas as pd

# Make backend/ and prompts/ importable when running pytest from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_df():
    """The same small dataset used throughout manual testing in this project."""
    return pd.DataFrame({
        "date": ["2024-01-15", "2024-01-18", "2024-02-01", "2024-02-14", "2024-03-10"],
        "product": ["Laptop", "Phone", "Laptop", "Headphones", "Phone"],
        "region": ["North", "South", "South", "East", "North"],
        "revenue": [75000, 45000, 68000, 12000, 51000],
        "quantity": [15, 30, 13, 40, 34],
    })


@pytest.fixture
def csv_bytes_factory():
    """Returns a function that turns a CSV string into a file-like
    object matching what Streamlit's file_uploader hands to load_csv()."""
    def _make(csv_text: str, encoding: str = "utf-8"):
        return io.BytesIO(csv_text.encode(encoding))
    return _make
