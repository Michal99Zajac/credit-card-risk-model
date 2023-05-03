import numpy as np
import pandas as pd

from src.preprocessing.insert_nulls import insert_nulls


def test_insert_nulls():
    """
    Test the insert_nulls function to ensure it properly inserts null values into a DataFrame.

    This test will:
    - Create a sample DataFrame.
    - Call the insert_nulls function with specific parameters.
    - Check if the output DataFrame has the correct number of null values inserted.
    - Check if no row contains more than one null value.
    - Check if excluded_columns are not altered.
    """
    # Create a sample DataFrame
    data = {
        "A": np.random.randint(-100, 100, size=1000),
        "B": np.random.randint(-100, 100, size=1000),
        "C": np.random.randint(-100, 100, size=1000),
    }
    df = pd.DataFrame(data)

    # Call the insert_nulls function
    percentage = 0.4
    excluded_columns = ["C"]
    result_df = insert_nulls(df, percentage, excluded_columns)

    # Check if the output DataFrame has the correct number of null values inserted
    null_count = result_df.isnull().sum().sum()
    expected_null_count = int(len(df) * percentage)
    assert (
        null_count == expected_null_count
    ), f"Expected {expected_null_count} null values, but got {null_count}"

    # Check if no row contains more than one null value
    for _, row in result_df.iterrows():
        assert row.isnull().sum() <= 1, "A row contains more than one null value"

    # Check if excluded_columns are not altered
    assert not result_df["C"].isnull().any(), "Null values found in excluded column"
