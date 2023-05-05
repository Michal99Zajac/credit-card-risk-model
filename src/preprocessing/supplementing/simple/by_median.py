from pandas import DataFrame


def by_median(df: DataFrame, columns: list[str]):
    """
    Fill missing values in specified columns of a DataFrame with the median of each column.

    Args:
        df (pandas.DataFrame): A Pandas DataFrame containing the data.
        columns (list[str]): A list of column names in the DataFrame to be filled with the median.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled in specified columns using the median.
    """
    for column in columns:
        # Calculate the median of the current column
        median = df[column].median(skipna=True)

        # Fill missing values with the median
        df[column].fillna(median, inplace=True)

    return df
