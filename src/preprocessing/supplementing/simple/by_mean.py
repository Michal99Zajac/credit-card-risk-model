from pandas import DataFrame


def by_mean(df: DataFrame, columns: list[str]):
    """
    Fill missing values in specified columns of a DataFrame with the mean of each column.

    Args:
        df (pandas.DataFrame): A Pandas DataFrame containing the data.
        columns (list[str]): A list of column names in the DataFrame to be filled with the mean.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled in specified columns using the mean.
    """
    for column in columns:
        # Calculate the mean of the current column
        mean = df[column].mean(skipna=True)

        # Fill missing values with the mean
        df[column].fillna(mean, inplace=True)

    return df
