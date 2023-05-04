import pandas as pd

from src.preprocessing.converting.to_binary_columns import to_binary_columns


def test_to_binary_columns():
    """
    Test the to_binary_columns function to ensure it properly converts a categorical Series into binary columns.

    This test will:
    - Create a pandas Series with some sample data.
    - Call the to_binary_columns function on the sample data.
    - Create the expected DataFrame with binary columns.
    - Assert that the result DataFrame is equal to the expected DataFrame.
    """
    # Create a pandas Series with some sample data
    sample_data = pd.Series(["a", "b", "c", "a", "b", "c"], name="category")

    # Apply the to_binary_columns function on the sample data
    result = to_binary_columns(sample_data)

    # Create the expected DataFrame with binary columns
    expected_data = {
        "has_a": [1, 0, 0, 1, 0, 0],
        "has_b": [0, 1, 0, 0, 1, 0],
        "has_c": [0, 0, 1, 0, 0, 1],
    }
    expected_result = pd.DataFrame(expected_data)

    # Assert that the result DataFrame is equal to the expected DataFrame
    pd.testing.assert_frame_equal(result, expected_result)