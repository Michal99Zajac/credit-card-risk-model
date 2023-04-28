import pandas as pd
import pytest


@pytest.mark.parametrize(
    "csv_file, percentage",
    [
        (
            "./data/credit_card_approval.csv",
            0.1,
        )
    ],
)
def test_percentage_of_nulls(csv_file, percentage):
    """
    Test if the actual number of rows with exactly one null value matches the expected number.

    Args:
        csv_file (str): Path to the input CSV file.
        percentage (float): The expected percentage of rows with one null value.

    Raises:
        AssertionError: If the actual number of rows with one null value does not match the expected number.
    """
    # Read the csv file into a DataFrame
    data = pd.read_csv(csv_file)

    # Calculate the total number of rows in the DataFrame
    total_rows = len(data)

    # Create a boolean mask where True represents rows with exactly one null value
    null_rows = data.isnull().sum(axis=1) == 1

    # Calculate the expected number of rows with one null value based on the given percentage
    expected_null_rows = int(total_rows * percentage)

    # Calculate the actual number of rows with one null value using the boolean mask
    actual_null_rows = null_rows.sum()

    # Assert that the actual number of rows with one null value matches the expected number
    assert (
        actual_null_rows == expected_null_rows
    ), f"Expected {expected_null_rows} rows with one null value, but found {actual_null_rows}"
