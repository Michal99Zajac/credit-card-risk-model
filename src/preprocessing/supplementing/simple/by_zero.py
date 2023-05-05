from pandas import DataFrame


def by_zero(df: DataFrame, columns: list[str]):
    """
    Fill missing values in specified columns of a DataFrame with the 0 of each column.

    Args:
        df (pandas.DataFrame): A Pandas DataFrame containing the data.
        columns (list[str]): A list of column names in the DataFrame to be filled with the 0.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled in specified columns using 0.
    """
    for column in columns:
        # Fill missing values with the mean
        df[column].fillna(0, inplace=True)

    return df
