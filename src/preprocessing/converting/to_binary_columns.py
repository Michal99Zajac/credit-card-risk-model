import pandas as pd


def to_binary_columns(series, prefix="has"):
    """
    Creates binary columns for the unique values in the input pandas Series.

    Args:
        series (pd.Series): The input pandas Series containing categorical values.
        prefix (str, optional): The prefix for the new binary columns. Default is 'has'.

    Returns:
        pd.DataFrame: A DataFrame with binary columns for each unique value in the input Series.
    """
    # Create dummy variables for each unique value in the input Series.
    # The new binary columns will have names in the format 'prefix value'.
    dummies = pd.get_dummies(series, prefix=prefix, prefix_sep="_", dtype=int)

    # Return the DataFrame with binary columns.
    return dummies
