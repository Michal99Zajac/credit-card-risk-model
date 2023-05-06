import pandas as pd
from sklearn.preprocessing import LabelEncoder


def to_enum(df: pd.DataFrame, column_to_encode: str):
    """
    Encode a categorical column of a given Pandas DataFrame using the LabelEncoder.

    This function uses the `LabelEncoder` to transform the values in the specified `column_to_encode` to numerical
    labels. It returns a new DataFrame with the encoded column and the original DataFrame with the specified column
    dropped. It also returns the `LabelEncoder` object used to encode the column.

    Args:
        df (pandas.DataFrame): The DataFrame to encode.
        column_to_encode (str): The name of the column to encode.

    Returns:
        tuple: A tuple of two pandas.DataFrames and a LabelEncoder object. The first DataFrame is the original DataFrame
            with the specified column dropped and the encoded column added. The second DataFrame is a DataFrame
            containing only the encoded column. The LabelEncoder object can be used to inverse_transform the labels
            back to their original values.
    """
    # Create a new LabelEncoder object
    encoder = LabelEncoder()

    # Fit the encoder to the column to be encoded
    encoder.fit(df[column_to_encode])

    # Encode the column and store the result in a new DataFrame
    encoded_column = encoder.transform(df[column_to_encode])
    df[column_to_encode] = encoded_column

    # Return the encoded column DataFrame, and the LabelEncoder object
    return df, encoder
