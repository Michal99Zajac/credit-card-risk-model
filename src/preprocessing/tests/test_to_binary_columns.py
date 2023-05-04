import pandas as pd

from src.preprocessing.converting.to_binary_columns import to_binary_columns


def test_to_binary_columns():
    """
    Test the to_binary_columns function to ensure it properly converts a categorical DataFrame into binary columns.

    This test will:
    - Create a pandas DataFrame with some sample data.
    - Call the to_binary_columns function on the sample data.
    - Create the expected DataFrame with binary columns.
    - Assert that the result DataFrame is equal to the expected DataFrame.
    """
    # Create a pandas DataFrame with some sample data
    sample_data = pd.DataFrame({"category": ["a", "b", "c", "a", "b", "c"]})

    # Apply the to_binary_columns function on the sample data
    result = to_binary_columns(sample_data, ["category"])[0]

    # Create the expected DataFrame with binary columns
    expected_data = {
        "category_a": [1, 0, 0, 1, 0, 0],
        "category_b": [0, 1, 0, 0, 1, 0],
        "category_c": [0, 0, 1, 0, 0, 1],
    }
    expected_result = pd.DataFrame(expected_data)

    # Assert that the result DataFrame is equal to the expected DataFrame
    pd.testing.assert_frame_equal(result, expected_result)
