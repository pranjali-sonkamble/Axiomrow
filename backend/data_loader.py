# backend/data_loader.py

import logging
import os

import pandas as pd
import numpy as np

logger = logging.getLogger("axiomrow.data_loader")

# Enforced WHILE parsing (nrows), not after: a small file can expand to a
# very large frame, and parsing everything before checking is the DoS.
MAX_ROWS = 3_000_000
MAX_COLS = 500


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Takes a file uploaded via Streamlit and returns a pandas DataFrame.

    Handles the edge cases that used to crash or silently misbehave:
    - Empty files (0 rows or 0 bytes)
    - Non-UTF-8 encoded files (falls back to latin-1)
    - Duplicate column names (deduplicated with numeric suffixes,
      since pandas allows them but downstream code assumes unique names)
    """
    try:
        def _read(encoding=None):
            kw = {"encoding": encoding} if encoding else {}
            uploaded_file.seek(0)
            header = pd.read_csv(uploaded_file, nrows=0, **kw)
            if header.shape[1] > MAX_COLS:
                raise ValueError(
                    f"The CSV has {header.shape[1]:,} columns; the limit is {MAX_COLS}."
                )
            uploaded_file.seek(0)
            frame = pd.read_csv(uploaded_file, nrows=MAX_ROWS + 1, **kw)
            if len(frame) > MAX_ROWS:
                raise ValueError(f"The CSV has more than {MAX_ROWS:,} rows; that is over the limit.")
            return frame

        try:
            df = _read()
        except UnicodeDecodeError:
            # Common with exports from Excel on Windows — retry with a
            # more permissive encoding instead of failing outright.
            df = _read("latin-1")

        if df.shape[0] == 0:
            raise ValueError(
                "The CSV file has no data rows (only headers, or completely empty)."
            )
        if df.shape[1] == 0:
            raise ValueError("The CSV file has no columns.")

        if df.columns.duplicated().any():
            df.columns = _deduplicate_columns(df.columns)

        return df

    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Could not parse the CSV — it may be malformed: {str(e)[:200]}")
    except ValueError:
        raise  # re-raise our own messages above as-is
    except Exception:
        # Library/OS exception text can contain paths and internals.
        logger.exception("Unexpected CSV read failure")
        raise ValueError("Could not read the CSV file. Please check that it is a valid CSV.")



def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load a supported Axiomrow upload (CSV or XLSX) into a DataFrame."""
    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith((".xlsx", ".xls")):
        # .xlsx is a zip archive: a few KB can decompress to gigabytes.
        # The app only accepts CSV, so this path is OFF unless explicitly
        # enabled (and then still subject to the row/column limits below).
        if os.getenv("AXIOMROW_ENABLE_XLSX") != "1":
            raise ValueError("Excel uploads are not supported. Please upload a CSV.")
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {e}")
        if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
            raise ValueError("The Excel file is over the row/column limit.")
        if df.shape[0] == 0:
            raise ValueError("The Excel file has no data rows.")
        if df.shape[1] == 0:
            raise ValueError("The Excel file has no columns.")
        if df.columns.duplicated().any():
            df.columns = _deduplicate_columns(df.columns)
        return df
    return load_csv(uploaded_file)

def _deduplicate_columns(columns) -> list:
    """Turns ['a', 'a', 'b'] into ['a', 'a_2', 'b'] so every column
    name is unique — pandas allows duplicate column names, but every
    downstream function here assumes df[col] returns a single Series."""
    seen = {}
    result = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result


def get_data_profile(df: pd.DataFrame, max_detailed_columns: int = 40) -> dict:
    """
    Returns a dictionary with stats about the dataframe.

    PERFORMANCE NOTES (added after testing with real, larger datasets):
    - value_counts() is skipped for high-cardinality columns (e.g. ID
      columns, free-text) since they're expensive to compute AND not
      meaningful to show "top values" for. A column where more than
      50% of values are unique is treated as an identifier, not a
      real category.
    - Only the first `max_detailed_columns` get a full per-column
      breakdown. Datasets with many more columns than that get a
      summary note instead of scanning every single column, so
      profiling stays fast on wide datasets.
    - This function should be called ONCE per uploaded file and its
      result cached (e.g. in st.session_state), not recomputed on
      every Streamlit rerun — recomputing per-interaction is what
      caused the app to feel slow on larger datasets.
    """
    row_count = len(df)
    columns = list(df.columns)

    profile = {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": [],
        "missing_values_total": int(df.isnull().sum().sum()),
        "sample_rows": df.head(2).to_dict(orient="records"),
        "truncated_columns": max(0, len(columns) - max_detailed_columns),
    }

    columns_to_detail = columns[:max_detailed_columns]

    for col in columns_to_detail:
        series = df[col]
        col_info = {
            "name": col,
            "dtype": str(series.dtype),
            "missing_count": int(series.isnull().sum()),
            "unique_count": int(series.nunique()),
        }

        # Checked FIRST, for every dtype, not just numeric: an all-null
        # column loaded from CSV (or created with plain `None` values)
        # is dtype "object" in pandas, not "float64" — a column of pure
        # None does not upcast to float64 the way a column mixing None
        # with real numbers does. Gating "all_missing" behind a numeric
        # dtype check (as this used to be) meant that common case was
        # silently missed: it fell through to the categorical branch
        # below and produced an empty, unexplained top_values dict
        # instead of clearly saying the column has no usable values.
        if series.notna().sum() == 0:
            col_info["stats"] = None
            col_info["all_missing"] = True

        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            col_info["stats"] = {
                "min": float(series.min()) if row_count else 0,
                "max": float(series.max()) if row_count else 0,
                "mean": round(float(series.mean()), 2) if row_count else 0,
                "median": float(series.median()) if row_count else 0,
            }

        # Use is_numeric_dtype (not a literal "object" dtype check) so
        # this keeps working across pandas versions — pandas 3.0+
        # defaults string columns to a native "str" dtype instead of
        # "object", which the old literal check silently missed.
        elif not pd.api.types.is_numeric_dtype(series):
            # Skip value_counts() on likely-identifier columns — expensive
            # to compute and not meaningful (e.g. a customer_id column
            # where almost every value is unique).
            uniqueness_ratio = col_info["unique_count"] / row_count if row_count else 0
            if uniqueness_ratio > 0.5:
                col_info["is_identifier_like"] = True
            else:
                top_values = series.value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"].append(col_info)

    return profile


def get_llm_context(df: pd.DataFrame, profile: dict = None) -> str:
    """
    Creates a text summary of the dataframe to send to the LLM.

    Accepts an optional pre-computed `profile` dict (from
    get_data_profile) so the caller can reuse a cached profile instead
    of triggering a second full scan of the dataframe — this halves
    the work done per chat message on larger datasets.
    """
    if profile is None:
        profile = get_data_profile(df)

    lines = [
        f"Dataset has {profile['row_count']} rows and {profile['column_count']} columns.",
        f"Total missing values: {profile['missing_values_total']}",
        "",
        "Columns:"
    ]

    for col in profile["columns"]:
        line = f"- {col['name']} ({col['dtype']}): {col['unique_count']} unique values"
        if col["missing_count"] > 0:
            line += f", {col['missing_count']} missing"
        if col.get("stats"):
            s = col["stats"]
            line += f" | range: {s['min']} to {s['max']}, mean: {s['mean']}"
        if col.get("is_identifier_like"):
            line += " | (identifier-like column, values mostly unique)"
        elif "top_values" in col:
            top = list(col["top_values"].keys())[:3]
            line += f" | sample values: {', '.join(top)}"
        lines.append(line)

    if profile.get("truncated_columns", 0) > 0:
        lines.append(
            f"... and {profile['truncated_columns']} more columns not detailed here "
            f"(dataset is wide — ask about a specific column by name if needed)."
        )

    # Raw sample rows are real records (names, emails, IDs) and are only
    # regex-masked before leaving for the LLM provider. Set
    # AXIOMROW_SEND_SAMPLE_ROWS=0 for sensitive data: column names, types
    # and aggregate stats are still sent; raw records are not.
    if os.getenv("AXIOMROW_SEND_SAMPLE_ROWS", "1") != "0":
        lines.append("")
        lines.append("Sample rows:")
        for i, row in enumerate(profile["sample_rows"]):
            lines.append(f"Row {i+1}: {row}")

    return "\n".join(lines)