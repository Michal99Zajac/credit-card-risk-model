import pandas as pd

from src.preprocessing.converting.to_enum import to_enum

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


def test_to_enum():
    """
    Test the to_enum function to ensure it properly maps values to enum.

    This test will:
    - Define the expected transformed data as a pandas dataframe.
    - Apply the to_enum function to the original data using the value mapping.
    - Assert that the transformed data matches the expected transformed data.
    """
    # Define the expected transformed data as a pandas dataframe
    expected_transformed_data = pd.DataFrame({"edu": [[0, 1, 2, 3, 4, 1, 1, 4, 3]]})

    # Apply the to_enum function to the original data
    transformed_data = to_enum(original_data, "edu")[0]

    # Assert that the transformed data matches the expected transformed data
    assert transformed_data.equals(
        transformed_data
    ), f"Expected {expected_transformed_data}, but got {transformed_data}"
