"""Tests for backend/data_loader.py — CSV loading, profiling, and the
LLM-context text summary that gets sent to the model."""

import io
import pytest
import pandas as pd
from backend.data_loader import load_csv, get_data_profile, get_llm_context, _deduplicate_columns


class TestLoadCSV:
    def test_loads_valid_csv(self, csv_bytes_factory):
        df = load_csv(csv_bytes_factory("a,b\n1,2\n3,4\n"))
        assert df.shape == (2, 2)

    def test_rejects_empty_file_with_only_headers(self, csv_bytes_factory):
        with pytest.raises(ValueError, match="no data rows"):
            load_csv(csv_bytes_factory("col1,col2\n"))

    def test_rejects_completely_empty_file(self, csv_bytes_factory):
        with pytest.raises(ValueError):
            load_csv(csv_bytes_factory(""))

    def test_falls_back_to_latin1_on_unicode_error(self):
        # é encoded in latin-1, which is NOT valid utf-8
        raw = "name,city\nJos\xe9,S\xe3o Paulo\n".encode("latin-1")
        df = load_csv(io.BytesIO(raw))
        assert df.loc[0, "name"] == "José"

    def test_pandas_native_dedup_produces_valid_dataframe(self, csv_bytes_factory):
        # pandas itself renames "name","name" -> "name","name.1" during
        # read_csv, before our own dedup logic ever runs
        df = load_csv(csv_bytes_factory("name,name,revenue\nA,B,100\n"))
        assert len(df.columns) == len(set(df.columns))  # all unique
        assert df.shape[1] == 3


class TestDeduplicateColumns:
    def test_no_duplicates_unchanged(self):
        assert _deduplicate_columns(["a", "b", "c"]) == ["a", "b", "c"]

    def test_duplicates_get_numeric_suffix(self):
        assert _deduplicate_columns(["a", "a", "a"]) == ["a", "a_2", "a_3"]


class TestGetDataProfile:
    def test_basic_profile_shape(self, sample_df):
        profile = get_data_profile(sample_df)
        assert profile["row_count"] == 5
        assert profile["column_count"] == 5
        assert profile["missing_values_total"] == 0

    def test_numeric_column_gets_stats(self, sample_df):
        profile = get_data_profile(sample_df)
        revenue_col = next(c for c in profile["columns"] if c["name"] == "revenue")
        assert revenue_col["stats"]["min"] == 12000
        assert revenue_col["stats"]["max"] == 75000

    def test_all_nan_numeric_column_does_not_crash(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": pd.Series([None, None, None], dtype=float)})
        profile = get_data_profile(df)
        b_col = next(c for c in profile["columns"] if c["name"] == "b")
        assert b_col["stats"] is None
        assert b_col["all_missing"] is True

    def test_high_cardinality_column_marked_as_identifier(self):
        df = pd.DataFrame({"id": [f"ID-{i}" for i in range(100)]})
        profile = get_data_profile(df)
        id_col = profile["columns"][0]
        assert id_col.get("is_identifier_like") is True
        assert "top_values" not in id_col

    def test_wide_dataset_truncates_detailed_columns(self):
        df = pd.DataFrame({f"col_{i}": [1, 2, 3] for i in range(50)})
        profile = get_data_profile(df, max_detailed_columns=10)
        assert len(profile["columns"]) == 10
        assert profile["truncated_columns"] == 40


class TestGetLLMContext:
    def test_produces_non_empty_summary(self, sample_df):
        context = get_llm_context(sample_df)
        assert "5 rows" in context
        assert "revenue" in context

    def test_accepts_precomputed_profile_without_rescanning(self, sample_df):
        profile = get_data_profile(sample_df)
        context = get_llm_context(sample_df, profile=profile)
        assert "revenue" in context

    def test_notes_truncated_columns_for_wide_datasets(self):
        df = pd.DataFrame({f"col_{i}": [1, 2, 3] for i in range(50)})
        profile = get_data_profile(df, max_detailed_columns=10)
        context = get_llm_context(df, profile=profile)
        assert "40 more columns" in context
