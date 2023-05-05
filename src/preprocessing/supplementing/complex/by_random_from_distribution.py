import numpy as np
from pandas import DataFrame


def by_random_from_distribution(df: DataFrame, columns: list[str]):
    """
    Fill missing values in specified columns of a DataFrame by randomly sampling from a probability distribution
    calculated from the non-missing values in those columns.

    This function calculates the probability distribution of the non-missing values in each specified column and uses
    the distribution to randomly sample values to fill in the missing values in that column.

    Args:
        df (pandas.DataFrame): A Pandas DataFrame containing the data.
        columns (list[str]): A list of column names in the DataFrame to be filled with random sampling.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled in specified columns using random sampling from a
        calculated probability distribution.
    """
    for column in columns:
        # Get the non-missing values
        not_missing = df[column].dropna()

        # Calculate the probability distribution of the non-missing values
        prob_distribution = not_missing.value_counts(normalize=True)

        # Fill missing values by sampling from the calculated distribution
        missing_values = df[column].isnull()
        df.loc[missing_values, column] = np.random.choice(
            prob_distribution.index, size=missing_values.sum(), p=prob_distribution.values
        )

    return df
