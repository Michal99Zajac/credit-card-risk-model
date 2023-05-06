import pandas as pd


def standard(data: pd.Series):
    """
    Scale the values in a Pandas Series using standardization.

    This function scales the values in the input Pandas Series to a standard normal distribution (mean=0, standard deviation=1)
    using standardization. The formula used for standardization is: (data - mean) / std_dev.

    Args:
        data (pandas.Series): A Pandas Series containing the data to be scaled.

    Returns:
        pandas.Series: The scaled data.
    """
    # Calculate the mean and standard deviation of the data
    mean = data.mean()
    std_dev = data.std()

    # Scale the data using standardization
    scaled_data = (data - mean) / std_dev

    return scaled_data
