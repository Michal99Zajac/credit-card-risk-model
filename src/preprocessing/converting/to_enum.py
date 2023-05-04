import numpy as np
import pandas as pd


def to_enum(data, mapping):
    """
    Transforms the input data based on the given value mapping.

    Args:
        data (pd.Series): A pandas Series containing the original data to be transformed.
        mapping (dict): A dictionary defining the mapping between the original values
                        and the desired enum values.

    Returns:
        pd.Series: A pandas Series containing the transformed data.
    """
    # Create a Series with the same index as the original data, filled with NaN values
    enum_data = pd.Series(np.nan, index=data.index)

    # Apply the mapping to the original data
    for original_value, enum_value in mapping.items():
        enum_data[data == original_value] = enum_value

    # Check if there are any unmatched values and print a warning if found
    unmatched_values = data[enum_data.isna()].unique()
    if len(unmatched_values) > 0:
        print(f"Warning: Unmatched values found in the input data: {unmatched_values}")

    return enum_data
