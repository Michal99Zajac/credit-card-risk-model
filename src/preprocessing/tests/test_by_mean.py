import pandas as pd

from src.preprocessing.supplementing.simple.by_mean import by_mean


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
