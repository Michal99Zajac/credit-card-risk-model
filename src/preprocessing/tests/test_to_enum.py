import pandas as pd

from src.preprocessing.converting.to_enum import to_enum

# Define the original data as a pandas Series
original_data = pd.Series(
    [
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
)

# Define a dictionary mapping original values to new values
value_mapping = {
    "Secondary / secondary special": 1,
    "Higher education": 2,
    "Incomplete higher": 3,
    "Lower secondary": 4,
    "Academic degree": 5,
}


def test_to_enum():
    """
    Test the to_enum function to ensure it properly maps values to a specified enum.

    This test will:
    - Define the expected transformed data as a pandas Series.
    - Apply the to_enum function to the original data using the value mapping.
    - Assert that the transformed data matches the expected transformed data.
    """
    # Define the expected transformed data as a pandas Series
    expected_transformed_data = pd.Series([1, 2, 3, 4, 5, 2, 2, 5, 4])

    # Apply the to_enum function to the original data using the value mapping
    transformed_data = to_enum(original_data, value_mapping)

    # Assert that the transformed data matches the expected transformed data
    assert transformed_data.equals(
        transformed_data
    ), f"Expected {expected_transformed_data}, but got {transformed_data}"
