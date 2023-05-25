import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def insert_nulls(df, percentage=0.1, excluded_columns=None):
    """
    Insert null values into a given Pandas DataFrame such that each row contains at most one null value.

    This function randomly selects `percentage` of the rows and inserts a single null value into an eligible
    column of each selected row. It ensures that no row has more than one null value inserted and avoids
    overwriting existing null values.

    Args:
        df (pandas.DataFrame): The DataFrame to insert null values into.
        percentage (float): The percentage of rows to fill with null values. Default is 0.1.
        excluded_columns (list): A list of columns to exclude from having null values inserted. Default is None.

    Raises:
        ValueError: If there are no eligible columns to insert null values into.

    Returns:
        pandas.DataFrame: The input DataFrame with null values inserted.
    """
    if excluded_columns is None:
        excluded_columns = []

    # Calculate the number of rows to fill with null values
    total_rows = len(df)
    null_rows = int(total_rows * percentage)

    # Select columns where null values can be inserted
    eligible_columns = [col for col in df.columns if col not in excluded_columns]

    # Check if there are any columns eligible for inserting null values
    if not eligible_columns:
        raise ValueError("No eligible columns to insert null values")

    # Fill with null values
    null_inserted_rows = set()
    while len(null_inserted_rows) < null_rows:
        row_idx = random.randint(0, total_rows - 1)

        # Ensure we're not inserting a second null value in the same row
        if row_idx in null_inserted_rows:
            continue

        col_idx = random.choice(eligible_columns)

        # Check if the cell is not already a null value
        if pd.isnull(df.at[row_idx, col_idx]):
            continue

        df.at[row_idx, col_idx] = np.nan
        null_inserted_rows.add(row_idx)

    return df


if __name__ == "__main__":
    file_path = "../../db/credit_card_approval.csv"
    df = pd.read_csv(file_path)

    # Get reduced dataset
    _, df = train_test_split(df, stratify=df["TARGET"], test_size=0.1, random_state=42)
    df = df.reset_index(drop=True)

    # Exclude "ID" and "TARGET" columns from being filled with null values
    excluded_columns = [
        "ID",
        "TARGET",
    ]

    # Fill with null values
    df = insert_nulls(
        df,
        percentage=0.1,
        excluded_columns=excluded_columns,
    )

    # Save the data to a CSV file
    df.to_csv(file_path, index=False)
