import pandas as pd

from src.preprocessing.supplementing.simple.by_mean import by_mean
from src.preprocessing.supplementing.simple.by_median import by_median
from src.preprocessing.supplementing.simple.by_mode import by_mode
from src.preprocessing.supplementing.simple.by_zero import by_zero


def test_by_mean():
    """
    Test the by_mean function to ensure it properly fills missing values in a DataFrame using the mean.

    This test will:
    - Create a sample DataFrame with missing values.
    - Call the by_mean function with specific columns.
    - Check if the output DataFrame has the correct values filled in using the mean.
    - Check if the non-specified columns are not altered.
    """
    # Create a test DataFrame
    df = pd.DataFrame({"A": [1, 2, 3, 4, None, 5, None], "B": [None, 2, 2, 2, 1, 1, 1]})

    # Use the by_mean function to fill missing values
    by_mean(df, ["A", "B"])

    # Expected output after filling missing values
    expected_df = pd.DataFrame({"A": [1, 2, 3, 4, 3, 5, 3], "B": [1.5, 2, 2, 2, 1, 1, 1]})

    # Assert that the filled DataFrame is equal to the expected DataFrame
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_by_mode():
    """
    Test the by_mode function to ensure it properly fills missing values in a DataFrame using the mode.

    This test will:
    - Create a sample DataFrame with missing values.
    - Call the by_mode function with specific columns.
    - Check if the output DataFrame has the correct values filled in.
    - Check if the non-specified columns are not altered.
    """
    # Create a test DataFrame
    df = pd.DataFrame({"A": [1, 2, 3, 2, None, 2, None], "B": [None, 2, 2, 1, 1, 1, 1]})

    # Use the by_mode function to fill missing values
    by_mode(df, ["A", "B"])

    # Expected output after filling missing values
    expected_df = pd.DataFrame({"A": [1, 2, 3, 2, 2, 2, 2], "B": [1, 2, 2, 1, 1, 1, 1]})

    # Assert that the filled DataFrame is equal to the expected DataFrame
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_by_zero():
    """
    Test the by_zero function to ensure it properly fills missing values in a DataFrame with zero.

    This test will:
    - Create a sample DataFrame with missing values.
    - Call the by_zero function with specific columns.
    - Check if the output DataFrame has the correct values filled in with zero.
    - Check if the non-specified columns are not altered.
    """
    # Create a test DataFrame
    df = pd.DataFrame({"A": [1, 2, 3, 2, None, 2, None], "B": [None, 2, 2, 1, 1, 1, 1]})

    # Use the by_zero function to fill missing values
    by_zero(df, ["A", "B"])

    # Expected output after filling missing values
    expected_df = pd.DataFrame({"A": [1, 2, 3, 2, 0, 2, 0], "B": [0, 2, 2, 1, 1, 1, 1]})

    # Assert that the filled DataFrame is equal to the expected DataFrame
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_by_median():
    """
    Test the by_median function to ensure it properly fills missing values in a DataFrame with the median.

    This test will:
    - Create a sample DataFrame with missing values.
    - Call the by_median function with specific columns.
    - Check if the output DataFrame has the correct values filled in with the median.
    - Check if the non-specified columns are not altered.
    """
    # Create a test DataFrame
    df = pd.DataFrame({"A": [1, 2, 3, 2, None, 2, None], "B": [None, 2, 2, 1, 1, 1, 1]})

    # Use the by_median function to fill missing values
    by_median(df, ["A", "B"])

    # Expected output after filling missing values
    expected_df = pd.DataFrame({"A": [1, 2, 3, 2, 2, 2, 2], "B": [1, 2, 2, 1, 1, 1, 1]})

    # Assert that the filled DataFrame is equal to the expected DataFrame
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)
