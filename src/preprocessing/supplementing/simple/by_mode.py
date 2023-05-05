from pandas import DataFrame


def by_mode(df: DataFrame, columns: list[str]):
    """
    Fill missing values in specified columns of a DataFrame with the mode of each column.

    Args:
        df(pd.DataFrame): A Pandas DataFrame containing the data.
        columns(list[str]): A list of column names in the DataFrame to be filled with the mode.

    Returns:
        DataFrame: The DataFrame with missing values filled in specified columns.
    """
    for column in columns:
        # Calculate the mode of the current column
        mode = df[column].mode().iloc[0]

        # Fill missing values with the mode
        df[column].fillna(mode, inplace=True)

    return df
