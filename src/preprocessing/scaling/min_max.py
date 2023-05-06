import pandas as pd


def min_max(data: pd.Series, new_range: tuple[float, float]):
    """
    Scale the values in a Pandas Series to a new range using min-max normalization.

    This function scales the values in the input Pandas Series to a new range defined by `new_range` using min-max
    normalization. The formula used for min-max normalization is: (data - min_value) / (max_value - min_value) * (new_max - new_min) + new_min.

    Args:
        data (pandas.Series): A Pandas Series containing the data to be scaled.
        new_range (tuple[float, float]): A tuple containing the new range to scale the data to.

    Returns:
        pandas.Series: The scaled data.
    """
    # Get the minimum and maximum values from the data
    min_value = data.min()
    max_value = data.max()

    # Get the new minimum and maximum values
    new_min, new_max = new_range

    # Scale the data using min-max normalization
    scaled_data = (data - min_value) / (max_value - min_value) * (new_max - new_min) + new_min

    return scaled_data
