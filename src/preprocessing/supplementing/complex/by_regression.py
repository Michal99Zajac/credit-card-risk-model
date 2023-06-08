import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression


def by_regression(df: DataFrame, columns: list[str]):
    """
    Fill missing values in specified columns of a DataFrame using a linear regression model.

    This function uses linear regression to predict missing values in the specified columns based on the non-missing
    values in those columns. For each column, the function fits a linear regression model using the non-missing data,
    then uses the model to predict the missing values.

    Args:
        df (pandas.DataFrame): A Pandas DataFrame containing the data.
        columns (list[str]): A list of column names in the DataFrame to be filled with regression.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled in specified columns using regression.
    """
    for column in columns:
        # Get the indices of missing values and non-missing values
        missing = df[column].isnull()
        not_missing = ~missing

        # Fit a linear regression model using the non-missing data
        X = np.arange(len(df)).reshape(-1, 1)
        y = df.loc[not_missing, column].values
        X_not_missing = X[not_missing].reshape(-1, 1)

        model = LinearRegression()
        model.fit(X_not_missing, y)

        # Predict the missing values using the fitted model
        X_missing = X[missing].reshape(-1, 1)
        y_missing = model.predict(X_missing)

        # Fill in the missing values
        df.loc[missing, column] = y_missing

    return df
