from pandas import DataFrame
from sklearn.impute import KNNImputer


def by_knn(df: DataFrame, columns: list[str], n_neighbors: int = 3):
    """
    Fill missing values in specified columns of a DataFrame using k-nearest neighbors (kNN) imputation.

    This function uses the kNN algorithm to impute missing values in the specified columns of the input DataFrame.
    For each column, the function fits a kNN model using the non-missing data, then uses the model to impute the
    missing values.

    Args:
        df (pandas.DataFrame): A Pandas DataFrame containing the data.
        columns (list[str]): A list of column names in the DataFrame to be imputed using kNN.
        n_neighbors (int): The number of neighbors to use for the kNN algorithm. Default is 3.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled in specified columns using kNN imputation.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)

    for column in columns:
        # Reshape the column data into a 2D array
        column_data = df[column].values.reshape(-1, 1)

        # Fill missing values using kNN
        imputed_data = imputer.fit_transform(column_data)

        # Replace the original column with the imputed data
        df[column] = imputed_data.ravel()

    return df
