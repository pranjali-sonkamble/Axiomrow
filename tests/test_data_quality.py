import pandas as pd

from backend.data_quality import assess_data_quality


def test_clean_dataset_is_ready():
    df = pd.DataFrame({
        "customer": ["A", "B", "C", "D"],
        "revenue": [100, 200, 150, 175],
        "region": ["North", "South", "North", "West"],
    })
    result = assess_data_quality(df)
    assert result["analysis_ready"] is True
    assert result["status"] == "READY"


def test_missing_values_are_detected_without_automatically_blocking_small_amounts():
    df = pd.DataFrame({
        "customer": ["A", "B", None, "D", "E", "F", "G", "H", "I", "J"],
        "revenue": list(range(10)),
    })
    result = assess_data_quality(df)
    assert any(i["issue"] == "Missing values present" for i in result["info"])
    assert result["analysis_ready"] is True


def test_high_missing_column_blocks():
    df = pd.DataFrame({
        "customer": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "income": [None] * 8 + [100, 200],
    })
    result = assess_data_quality(df)
    assert result["analysis_ready"] is False
    assert any(i["issue"] == "Extremely high missing values" for i in result["blocking_issues"])


def test_duplicate_rows_are_detected():
    df = pd.DataFrame({
        "id": [1, 1, 2, 3, 4],
        "value": [10, 10, 20, 30, 40],
    })
    result = assess_data_quality(df)
    assert any(i["issue"] == "Duplicate records" for i in result["issues"])


def test_invalid_numeric_as_text_is_detected():
    values = [str(x) for x in range(25)] + ["bad", "bad", "bad"]
    df = pd.DataFrame({"amount": values})
    result = assess_data_quality(df)
    assert any(i["issue"] == "Numeric values stored as text / mixed numeric type"
               for i in result["issues"])


def test_invalid_dates_are_detected():
    values = ["2026-01-01"] * 25 + ["not-a-date"] * 5
    df = pd.DataFrame({"order_date": values, "amount": list(range(30))})
    result = assess_data_quality(df)
    assert any(i["issue"] == "Invalid / inconsistent date values"
               for i in result["issues"])


def test_empty_and_constant_columns_are_detected():
    df = pd.DataFrame({
        "empty": [None] * 30,
        "constant": ["same"] * 30,
        "value": list(range(30)),
    })
    result = assess_data_quality(df)
    assert any(i["issue"] == "Completely empty columns" for i in result["blocking_issues"])
    assert any(i["issue"] == "Constant / zero-variation columns" for i in result["issues"])


def test_multiple_quality_issues_are_reported():
    df = pd.DataFrame({
        "order_date": ["2026-01-01"] * 25 + ["bad"] * 5,
        "amount": [str(x) for x in range(25)] + ["bad"] * 5,
        "region": ["North"] * 30,
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = assess_data_quality(df)
    names = {i["issue"] for i in result["issues"]}
    assert "Invalid / inconsistent date values" in names
    assert "Numeric values stored as text / mixed numeric type" in names
    assert "Duplicate records" in names
    assert "Constant / zero-variation columns" in names


def test_zero_columns_blocks():
    df = pd.DataFrame(index=range(5))
    result = assess_data_quality(df)
    assert result["analysis_ready"] is False


def test_empty_dataset_blocks():
    df = pd.DataFrame(columns=["a", "b"])
    result = assess_data_quality(df)
    assert result["analysis_ready"] is False
