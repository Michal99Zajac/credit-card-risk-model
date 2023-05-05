import pandas as pd

from src.preprocessing.supplementing.simple.by_mode import by_mode


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
