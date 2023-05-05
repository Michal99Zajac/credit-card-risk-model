import pandas as pd

from src.preprocessing.supplementing.complex.by_regression import by_regression


def test_by_regression():
    """
    Test the by_regression function to ensure it properly fills missing values using linear regression.

    This test will:
    - Create a sample DataFrame with missing values.
    - Call the by_regression function with specific columns.
    - Check if the output DataFrame has the correct values filled in.
    - Check if the non-specified columns are not altered.
    """
    # Create a test DataFrame
    df = pd.DataFrame({"A": [1, 2, None, 4, None, 6], "B": [1, 2, 3, None, None, 6]})

    # Use the by_regression function to fill missing values
    filled_df = by_regression(df, ["A"])

    # Expected output after filling missing values
    expected_df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [1, 2, 3, None, None, 6]})

    # Assert that the filled DataFrame is similar to the expected DataFrame
    pd.testing.assert_frame_equal(filled_df.round(), expected_df, check_dtype=False)
