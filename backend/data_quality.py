"""
Axiomrow deterministic data-quality gate.

This module is intentionally read-only: it profiles/evaluates a DataFrame and
returns structured findings. It never mutates the supplied DataFrame.
"""

from __future__ import annotations

from typing import Any, Dict, List
import re

import numpy as np
import pandas as pd


DATE_NAME_HINTS = (
    "date", "time", "timestamp", "day", "month", "year",
    "dob", "birth", "created", "updated", "start", "end",
)

NUMERIC_NAME_HINTS = (
    "amount", "price", "revenue", "sales", "income", "salary",
    "quantity", "qty", "age", "score", "rate", "percent", "percentage",
    "discount", "cost", "profit", "balance",
)

def _issue(
    issue: str,
    severity: str,
    observed: str,
    why: str,
    action: str,
    *,
    columns: List[str] | None = None,
    affected_rows: int | None = None,
    affected_pct: float | None = None,
    category: str = "quality",
) -> Dict[str, Any]:
    return {
        "issue": issue,
        "severity": severity,
        "columns": columns or [],
        "affected_rows": affected_rows,
        "affected_pct": affected_pct,
        "observed": observed,
        "why": why,
        "recommended_action": action,
        "category": category,
    }


def _safe_pct(n: int | float, total: int) -> float:
    return round((float(n) / total) * 100, 2) if total else 0.0


def _is_date_like_name(name: str) -> bool:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
    return any(h in s.split("_") for h in DATE_NAME_HINTS) or any(
        h in s for h in ("date", "timestamp", "datetime")
    )


def _is_numeric_like_name(name: str) -> bool:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
    return any(h in s.split("_") for h in NUMERIC_NAME_HINTS) or any(
        h in s for h in ("revenue", "salary", "amount", "quantity", "price")
    )


def _detect_numeric_as_text(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> None:
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        non_missing = s.dropna()
        if len(non_missing) < 20:
            continue

        # Cheap probe: a column needs >=80% numeric-like values to be flagged, so a
        # strided sample that is mostly non-numeric rules it out without converting
        # millions of strings.
        probe = non_missing.iloc[::max(1, len(non_missing) // 2000)].head(2000)
        probe_ratio = float(pd.to_numeric(
            probe.astype("string").str.replace(",", "", regex=False).str.strip(),
            errors="coerce").notna().mean())
        if probe_ratio < 0.5:
            continue

        converted = pd.to_numeric(
            non_missing.astype("string").str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
        numeric_ratio = float(converted.notna().mean())

        # Only flag when the column strongly looks numeric. This avoids
        # classifying ordinary categorical text as a type error.
        if numeric_ratio >= 0.80 and numeric_ratio < 1.0:
            bad = int(converted.isna().sum())
            pct = _safe_pct(bad, len(non_missing))
            issues.append(_issue(
                "Numeric values stored as text / mixed numeric type",
                "BLOCKING" if pct >= 10 else "WARNING",
                f"{bad:,} of {len(non_missing):,} non-missing values ({pct:.1f}%) "
                "cannot be parsed as numeric while most values are numeric-like.",
                "Numeric calculations, aggregations and visualizations can fail or "
                "produce incomplete results.",
                "Convert valid values to a consistent numeric type and investigate "
                "or correct the non-numeric entries before analysis.",
                columns=[str(col)],
                affected_rows=bad,
                affected_pct=pct,
                category="data_type",
            ))


def _detect_invalid_dates(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> None:
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue

        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        non_missing = s.dropna()
        if len(non_missing) < 20:
            continue

        parsed = pd.to_datetime(non_missing, errors="coerce", format="mixed")
        parse_ratio = float(parsed.notna().mean())
        name_like = _is_date_like_name(col)

        # A date-like column name plus substantial parse success is enough to
        # identify a date field. We only block when a meaningful portion fails.
        if name_like and parse_ratio >= 0.50:
            bad = int(parsed.isna().sum())
            pct = _safe_pct(bad, len(non_missing))
            if bad:
                issues.append(_issue(
                    "Invalid / inconsistent date values",
                    "BLOCKING" if pct >= 10 else "WARNING",
                    f"{bad:,} of {len(non_missing):,} non-missing values ({pct:.1f}%) "
                    "could not be parsed consistently as dates.",
                    "Time-based grouping, sorting and trend analysis can become "
                    "incorrect or incomplete.",
                    "Standardize the date format, convert valid values to a "
                    "consistent datetime type, and investigate invalid entries.",
                    columns=[str(col)],
                    affected_rows=bad,
                    affected_pct=pct,
                    category="date",
                ))


def _detect_category_inconsistency(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> None:
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        if int(s.notna().sum()) < 20:
            continue
        # Fast exit on DISTINCT values only: if stripping/casefolding collapses
        # none of them there is nothing to report (identical outcome, no 1M-row string pass).
        _u = pd.Series(s.dropna().unique(), dtype="string").str.strip()
        if _u.nunique() <= _u.str.casefold().nunique():
            continue
        non_missing = s.dropna().astype("string").str.strip()

        # Detect values that become identical after trimming/case normalization,
        # e.g. "North", " north ", "NORTH".
        normalized = non_missing.str.casefold()
        duplicate_normalized = int(normalized.duplicated(keep=False).sum())
        if duplicate_normalized == 0:
            continue

        raw_unique = int(non_missing.nunique(dropna=True))
        normalized_unique = int(normalized.nunique(dropna=True))
        if raw_unique <= normalized_unique:
            continue

        affected_pct = _safe_pct(duplicate_normalized, len(non_missing))
        issues.append(_issue(
            "Inconsistent categorical values",
            "WARNING",
            f"{raw_unique:,} raw categories collapse to {normalized_unique:,} "
            f"after trimming/case normalization; {duplicate_normalized:,} values "
            f"({affected_pct:.1f}%) are potentially inconsistent variants.",
            "Grouping and aggregation may split one logical category into multiple "
            "labels and distort counts or totals.",
            "Standardize whitespace, capitalization and known category labels; "
            "review the variants before analysis.",
            columns=[str(col)],
            affected_rows=duplicate_normalized,
            affected_pct=affected_pct,
            category="category",
        ))


def _detect_outliers(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> None:
    # Bounded to avoid expensive full-column sorting/operations on very large data.
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 30 or s.nunique() < 5:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((s < lower) | (s > upper)).sum())
        pct = _safe_pct(outlier_count, len(s))

        if outlier_count and pct >= 1:
            issues.append(_issue(
                "Potential outliers",
                "WARNING",
                f"{outlier_count:,} values ({pct:.1f}% of non-missing numeric "
                f"values) fall outside the 1.5×IQR range.",
                "Extreme values can materially influence means, correlations, "
                "scales and visualizations; they may be valid or erroneous.",
                "Investigate the extreme records and confirm whether they are "
                "legitimate observations before deciding how to handle them.",
                columns=[str(col)],
                affected_rows=outlier_count,
                affected_pct=pct,
                category="outlier",
            ))



def _build_result(
    df: pd.DataFrame,
    issues: List[Dict[str, Any]],
    info: List[Dict[str, Any]],
) -> Dict[str, Any]:
    blocking = [x for x in issues if x["severity"] == "BLOCKING"]
    warnings = [x for x in issues if x["severity"] == "WARNING"]

    if blocking:
        status = "NOT READY"
        reason = (
            f"{len(blocking)} blocking data-quality issue(s) must be resolved "
            "before reliable analysis."
        )
    else:
        status = "READY"
        reason = (
            "No blocking data-quality issues were detected. The dataset meets "
            "the minimum quality gate for the existing analysis workflow."
        )

    cleaning_plan = []
    priority = 1
    for item in blocking + warnings:
        cleaning_plan.append({
            "priority": priority,
            "issue": item["issue"],
            "severity": item["severity"],
            "columns": item.get("columns", []),
            "action": item["recommended_action"],
            "reason": item["why"],
            "expected_result": _expected_result(item["issue"]),
        })
        priority += 1

    return {
        "analysis_ready": not bool(blocking),
        "status": status,
        "reason": reason,
        "issues": issues,
        "blocking_issues": blocking,
        "warnings": warnings,
        "info": info,
        "cleaning_plan": cleaning_plan,
        "profile_summary": _profile_summary(df),
    }


def _expected_result(issue_name: str) -> str:
    mapping = {
        "Duplicate records": "Prevent accidental inflation of counts, totals and frequencies.",
        "Missing values": "Improve completeness for analyses that depend on the affected fields.",
        "Extremely high missing values": "Ensure the affected dimension has enough usable observations.",
        "Invalid / inconsistent date values": "Enable reliable date parsing, sorting and time-based analysis.",
        "Numeric values stored as text / mixed numeric type": "Enable reliable numeric aggregation and visualization.",
        "Completely empty columns": "Remove unusable fields and simplify the analytical schema.",
        "Completely empty rows": "Prevent blank records from affecting row counts and completeness.",
        "Blank / placeholder column names": "Make field references unambiguous for analysis and generated queries.",
        "Duplicate column names": "Give every field a unique analytical reference.",
        "Non-finite numeric values": "Prevent invalid numeric values from breaking calculations and charts.",
    }
    return mapping.get(issue_name, "Improve the reliability and interpretability of downstream analysis.")


def _profile_summary(df: pd.DataFrame) -> Dict[str, Any]:
    numeric = int(len(df.select_dtypes(include=[np.number]).columns))
    categorical = int(len(df.select_dtypes(include=["object", "string", "category"]).columns))
    date_cols = int(sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns))
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_columns": date_cols,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated(keep="first").sum()) if len(df) else 0,
    }

def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a deterministic, explainable readiness assessment.

    Only BLOCKING findings make analysis_ready=False. Warnings and informational
    findings are surfaced but do not by themselves stop the existing pipeline.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The uploaded data could not be represented as a DataFrame.")

    rows, cols = df.shape
    issues: List[Dict[str, Any]] = []
    info: List[Dict[str, Any]] = []

    if cols == 0:
        issues.append(_issue(
            "No columns",
            "BLOCKING",
            "The dataset contains zero columns.",
            "The analysis pipeline requires a tabular schema with at least one field.",
            "Re-export the dataset with valid column headers and at least one column.",
        ))
        return _build_result(df, issues, info)

    if rows == 0:
        issues.append(_issue(
            "Empty dataset",
            "BLOCKING",
            "The dataset contains zero data rows.",
            "There are no observations from which to calculate statistics, charts or insights.",
            "Provide a dataset containing valid records.",
        ))
        return _build_result(df, issues, info)

    # Structural column-name checks.
    col_strings = [str(c) for c in df.columns]
    blank_cols = [c for c in col_strings if not c.strip() or c.strip().lower().startswith("unnamed")]
    duplicated_names = pd.Index(col_strings).duplicated(keep=False)
    duplicated_name_values = sorted(set(
        col_strings[i] for i, flag in enumerate(duplicated_names) if flag
    ))

    if blank_cols:
        issues.append(_issue(
            "Blank / placeholder column names",
            "BLOCKING",
            f"{len(blank_cols)} column(s) have blank or placeholder names.",
            "Unclear field names make downstream analysis and generated questions "
            "ambiguous and can cause collisions in query/chart logic.",
            "Rename the affected columns with unique, meaningful names.",
            columns=blank_cols,
        ))

    if duplicated_name_values:
        issues.append(_issue(
            "Duplicate column names",
            "BLOCKING",
            f"Duplicate column name(s): {', '.join(duplicated_name_values[:8])}"
            + ("…" if len(duplicated_name_values) > 8 else ""),
            "Duplicate field names make column references ambiguous.",
            "Rename duplicate columns so every field has a unique name.",
            columns=duplicated_name_values,
        ))

    # Empty rows / columns.
    empty_cols = [str(c) for c in df.columns if int(df[c].notna().sum()) == 0]
    if empty_cols:
        issues.append(_issue(
            "Completely empty columns",
            "BLOCKING",
            f"{len(empty_cols)} column(s) contain no non-missing values.",
            "These fields cannot contribute reliable analysis and can confuse "
            "automated profiling and query generation.",
            "Remove the empty columns or populate them from the source system.",
            columns=empty_cols,
        ))

    empty_row_count = int(df.isna().all(axis=1).sum())
    if empty_row_count:
        pct = _safe_pct(empty_row_count, rows)
        issues.append(_issue(
            "Completely empty rows",
            "BLOCKING" if pct >= 5 else "WARNING",
            f"{empty_row_count:,} rows ({pct:.1f}%) are completely empty.",
            "Empty records can distort row counts and completeness metrics.",
            "Remove accidental blank records from the source data.",
            affected_rows=empty_row_count,
            affected_pct=pct,
        ))

    # Missingness: don't block simply because any values are missing.
    missing_by_col = df.isna().sum()
    total_missing = int(missing_by_col.sum())
    high_missing_cols = [
        str(c) for c in df.columns
        if _safe_pct(int(missing_by_col[c]), rows) >= 50
    ]
    if high_missing_cols:
        for col in high_missing_cols:
            n = int(missing_by_col[col])
            pct = _safe_pct(n, rows)
            issues.append(_issue(
                "Extremely high missing values",
                "BLOCKING" if pct >= 70 else "WARNING",
                f"{n:,} of {rows:,} values ({pct:.1f}%) are missing.",
                "A field with very little usable data can make analyses involving "
                "that dimension unreliable.",
                "Review the source field and apply a domain-appropriate missing-data strategy.",
                columns=[col],
                affected_rows=n,
                affected_pct=pct,
                category="missing",
            ))
    elif total_missing:
        info.append({
            "issue": "Missing values present",
            "severity": "INFO",
            "observed": f"{total_missing:,} missing cells across "
                        f"{int((missing_by_col > 0).sum()):,} column(s).",
        })

    # Duplicate rows.
    duplicate_count = int(df.duplicated(keep="first").sum())
    if duplicate_count:
        pct = _safe_pct(duplicate_count, rows)
        severity = "BLOCKING" if pct >= 20 else "WARNING"
        issues.append(_issue(
            "Duplicate records",
            severity,
            f"{duplicate_count:,} duplicate row(s) ({pct:.1f}% of the dataset).",
            "Duplicate records can inflate counts, sums, frequencies and other "
            "downstream metrics if they are accidental duplicates.",
            "Verify whether repeated rows are legitimate business records; remove "
            "only confirmed accidental duplicates before re-uploading.",
            affected_rows=duplicate_count,
            affected_pct=pct,
            category="duplicate",
        ))

    # Non-finite numeric values.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        s = df[col]
        bad = int((~np.isfinite(s.dropna())).sum()) if len(s.dropna()) else 0
        if bad:
            pct = _safe_pct(bad, rows)
            issues.append(_issue(
                "Non-finite numeric values",
                "BLOCKING",
                f"{bad:,} non-finite value(s) detected in {str(col)}.",
                "Infinity/NaN-like numeric values can break aggregations and charts.",
                "Replace or correct non-finite source values using the appropriate domain rule.",
                columns=[str(col)],
                affected_rows=bad,
                affected_pct=pct,
                category="numeric",
            ))

    _detect_numeric_as_text(df, issues)
    _detect_invalid_dates(df, issues)
    _detect_category_inconsistency(df, issues)

    # Constant columns are generally not a hard failure unless there is no
    # meaningful variation anywhere in the dataset.
    constant_cols = [
        str(c) for c in df.columns
        if int(df[c].nunique(dropna=True)) <= 1
    ]
    if constant_cols:
        issues.append(_issue(
            "Constant / zero-variation columns",
            "WARNING",
            f"{len(constant_cols)} column(s) contain one or fewer distinct "
            "non-missing values.",
            "A constant field cannot explain variation and may be an accidental "
            "export artifact.",
            "Review the fields and remove irrelevant constant columns if appropriate.",
            columns=constant_cols,
            category="structure",
        ))

    # High-cardinality categorical fields are not inherently invalid.
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        non_missing = df[col].dropna()
        if len(non_missing) < 50:
            continue
        cardinality_ratio = float(non_missing.nunique() / len(non_missing))
        if cardinality_ratio >= 0.95:
            info.append({
                "issue": "High-cardinality field",
                "severity": "INFO",
                "columns": [str(col)],
                "observed": f"{int(non_missing.nunique()):,} unique values among "
                            f"{len(non_missing):,} non-missing records "
                            f"({cardinality_ratio:.1%} unique).",
            })

    # Suspicious negative values are warnings only; they can be valid depending
    # on the domain (refunds, losses, temperature, etc.).
    for col in numeric_cols:
        if not _is_numeric_like_name(col):
            continue
        s = df[col].dropna()
        if len(s) and int((s < 0).sum()):
            neg = int((s < 0).sum())
            pct = _safe_pct(neg, len(s))
            issues.append(_issue(
                "Potentially invalid negative values",
                "WARNING",
                f"{neg:,} negative value(s) ({pct:.1f}%) detected in a field "
                "whose name suggests a non-negative measure.",
                "Negative values may be legitimate (for example refunds or losses) "
                "or may indicate data-entry problems.",
                "Validate the business rule and correct values only when the source "
                "definition confirms they are invalid.",
                columns=[str(col)],
                affected_rows=neg,
                affected_pct=pct,
                category="validity",
            ))

    _detect_outliers(df, issues)

    return _build_result(df, issues, info)
