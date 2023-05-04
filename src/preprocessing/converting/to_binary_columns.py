import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def to_binary_columns(df, columns_to_encode):
    """
    Encode categorical columns of a Pandas DataFrame as binary columns.

    This function uses the OneHotEncoder class from scikit-learn to encode each column in `columns_to_encode`
    as a set of binary columns in the output DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to encode.
        columns_to_encode (list): A list of column names to encode.

    Returns:
        pandas.DataFrame: A new DataFrame containing only the specified columns encoded as binary columns.
    """
    # Create an instance of the OneHotEncoder class
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int)

    # Fit the encoder to the specified columns
    encoder.fit(df[columns_to_encode])

    # Transform the specified columns into binary columns
    encoded_columns = encoder.transform(df[columns_to_encode])

    # Create a new DataFrame with the binary columns and their names
    binary_df = pd.DataFrame(
        encoded_columns, columns=encoder.get_feature_names_out(columns_to_encode)
    )

    # Return the new DataFrame with the binary columns
    return binary_df
