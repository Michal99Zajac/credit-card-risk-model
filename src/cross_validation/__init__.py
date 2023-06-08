import numpy as np
import pandas as pd


def stratified_cross_validation(df, target, n_folds=5):
    """
    Perform stratified k-fold cross-validation on a DataFrame.

    This function implements stratified k-fold cross-validation. The function begins by calculating the number of
    instances per class for each fold, given the number of unique classes in the target column. Then, it shuffles
    the DataFrame to ensure randomness.

    It subsequently creates stratified folds by iterating over the unique classes in the target column, shuffling
    the data for each class, and distributing it across the folds.

    The data within each fold is then combined, and the function concludes by splitting these folds into training
    and testing sets, ensuring that each fold is used as a test set exactly once.

    The function then returns a list of tuples, where each tuple contains a training DataFrame and a corresponding
    test DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame on which to conduct stratified k-fold cross-validation.
        target (str): The column used as the target variable.
        n_folds (int, optional): The number of folds. Defaults to 5.

    Returns:
        list[tuple[pandas.DataFrame, pandas.DataFrame]]: A list of tuples, where each tuple is a pair of training
        and test sets."""

    # Get unique target classes and counts
    classes, counts = np.unique(df[target], return_counts=True)

    # Shuffle the dataframe to ensure randomness
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate the number of instances of each class per fold
    n_per_class_per_fold = (counts / n_folds).astype(int)

    # Create stratified folds
    folds = [[] for _ in range(n_folds)]
    for class_index, c in enumerate(classes):
        # Shuffle the data for the class
        data_for_class = df[df[target] == c]

        # Distribute the data among the folds
        start = 0
        for i in range(n_folds):
            end = start + n_per_class_per_fold[class_index]
            folds[i].append(data_for_class.iloc[start:end])
            start = end

    # Combine the data in each fold
    for i in range(n_folds):
        folds[i] = pd.concat(folds[i], ignore_index=True)

    # Split folds into training and testing sets
    train_test_splits = []
    for i in range(n_folds):
        test = folds[i]
        train = pd.concat([folds[j] for j in range(n_folds) if j != i], ignore_index=True)
        train_test_splits.append((train, test))

    # Return the train/test splits
    return train_test_splits


def validate(train_test_splits, model, target):
    """
    Validate a model using the provided training and testing splits.

    This function takes a list of training and testing splits and a machine learning model. For each split, it fits
    the model using the training data and evaluates the performance using the test data. The performance is
    determined using the model's `score` method. The scores from all iterations are collected and returned as a list.

    It is assumed that the DataFrame splits provided have a column corresponding to the target variable and that
    the model provided has `fit` and `score` methods following the scikit-learn API.

    Args:
        train_test_splits (list[tuple[pandas.DataFrame, pandas.DataFrame]]): A list of tuples, each containing a
            training set and a testing set as DataFrames.
        model: The machine learning model to be validated. The model should have `fit` and `score` methods.
        target (str): The column used as the target variable.

    Returns:
        list[float]: A list of scores for each training/testing split."""

    # Perform cross-validation
    scores = []
    for train_test_split in train_test_splits:
        train, test = train_test_split

        # Fit the model
        model.fit(train.drop(columns=[target]), train[target])

        # Evaluate the model
        score = model.score(test.drop(columns=[target]), test[target])

        # Store the score
        scores.append(score)

    # Return the scores
    return scores
