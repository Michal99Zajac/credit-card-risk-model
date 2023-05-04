import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def to_binary_columns(df, columns_to_encode):
    """
    Encode specified categorical columns of a given Pandas DataFrame using the OneHotEncoder.

    This function uses the `OneHotEncoder` to transform the values in the specified `columns_to_encode` to binary
    columns. It returns a new DataFrame with the binary columns and the original DataFrame with the specified columns
    dropped. It also returns the `OneHotEncoder` object used to encode the columns.

    Args:
        df (pandas.DataFrame): The DataFrame to encode.
        columns_to_encode (list): A list of column names to encode.

    Returns:
        tuple: A tuple of two pandas.DataFrames and a OneHotEncoder object. The first DataFrame is the original DataFrame
            with the specified columns dropped and the binary columns added. The second DataFrame is a DataFrame
            containing only the binary columns. The OneHotEncoder object can be used to inverse_transform the binary
            columns back to their original values.
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
    return pd.concat([df.drop(columns_to_encode, axis=1), binary_df], axis=1), encoder
