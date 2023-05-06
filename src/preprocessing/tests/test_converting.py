import pandas as pd

from src.preprocessing.converting.to_binary_columns import to_binary_columns
from src.preprocessing.converting.to_enum import to_enum


def test_to_enum():
    """
    Test the to_enum function to ensure it properly maps values to enum.

    This test will:
    - Define the expected transformed data as a pandas dataframe.
    - Apply the to_enum function to the original data using the value mapping.
    - Assert that the transformed data matches the expected transformed data.
    """
    # Define the original data as a pandas dataframe
    original_data = pd.DataFrame(
        {
            "edu": [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
                "Higher education",
                "Higher education",
                "Academic degree",
                "Lower secondary",
            ]
        }
    )

    # Define the expected transformed data as a pandas dataframe
    expected_transformed_data = pd.DataFrame({"edu": [[0, 1, 2, 3, 4, 1, 1, 4, 3]]})

    # Apply the to_enum function to the original data
    transformed_data = to_enum(original_data, "edu")[0]

    # Assert that the transformed data matches the expected transformed data
    assert transformed_data.equals(
        transformed_data
    ), f"Expected {expected_transformed_data}, but got {transformed_data}"


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
