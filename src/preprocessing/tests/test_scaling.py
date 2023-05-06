import pandas as pd

from src.preprocessing.scaling.min_max import min_max
from src.preprocessing.scaling.standard import standard


def test_min_max():
    """
    Test the min_max function to ensure it properly scales data with a custom range.

    This test will:
    - Create a sample Series with values.
    - Call the min_max function with the sample Series and a custom range.
    - Check if the output Series has the correct scaled values.
    """
    # Create a test Series
    data = pd.Series([1, 2, 3, 4, 5])

    # Scale the data using custom range Min-Max scaling
    custom_range = (-1, 1)
    custom_range_scaled_data = min_max(data, custom_range)

    # Expected output after scaling
    expected_custom_range_scaled_data = pd.Series([-1, -0.5, 0, 0.5, 1])

    # Assert that the scaled Series is equal to the expected Series (up to a certain decimal point)
    pd.testing.assert_series_equal(custom_range_scaled_data, expected_custom_range_scaled_data)


def test_standard():
    """
    Test the standard function to ensure it properly scales data using standardization.

    This test will:
    - Create a sample Series with values.
    - Call the standard function with the sample Series.
    - Check if the output Series has the correct scaled values.
    """
    # Create a test Series
    data = pd.Series([1, 2, 3, 4, 5])

    # Scale the data using standardization
    standardized_data = standard(data)

    # Expected output after scaling
    expected_standardized_data = pd.Series([-1.26491106, -0.63245553, 0, 0.63245553, 1.26491106])

    # Assert that the scaled Series is equal to the expected Series (up to a certain decimal point)
    pd.testing.assert_series_equal(standardized_data, expected_standardized_data)
