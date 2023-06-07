import pandas as pd


def stratified_cross_validation(df, stratify_column, target_column, k=10):
    """
    Conduct stratified k-fold cross-validation on a DataFrame.

    This function performs stratified k-fold cross-validation on the given DataFrame. Stratification is done based
    on the unique classes in the `stratify_column`. This ensures that each fold is representative of the overall
    class distribution. The function shuffles the data within each class before creating the folds.

    Each fold consists of a training set and a test set, both of which maintain the original class distribution.
    The training and test sets are then split into their features (X) and target (y) components.

    Args:
        df (pandas.DataFrame): The DataFrame on which to conduct stratified k-fold cross-validation.
        stratify_column (str): The column used for stratification.
        target_column (str): The column used as the target variable.
        k (int, optional): The number of folds. Defaults to 10.

    Returns:
        list[list[pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series]]: A list of data splits, where each
        split is a list consisting of: training data (X_train), test data (X_test), training target (y_train), and test target (y_test).
    """
    unique_classes = df[stratify_column].unique()
    data_splits = []

    for fold in range(k):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for unique_class in unique_classes:
            class_data = df[df[stratify_column] == unique_class]
            class_data = class_data.sample(frac=1).reset_index(drop=True)  # shuffle

            size_of_fold = int(len(class_data) / k)

            if fold == k - 1:  # if it's the last fold
                test_fold = class_data[fold * size_of_fold :]
                train_fold = class_data[: fold * size_of_fold]
            else:
                test_fold = class_data[fold * size_of_fold : (fold + 1) * size_of_fold]
                train_fold = pd.concat(
                    [class_data[: fold * size_of_fold], class_data[(fold + 1) * size_of_fold :]]
                )

            train_data = pd.concat([train_data, train_fold])
            test_data = pd.concat([test_data, test_fold])

        X_train = train_data.drop(target_column, axis=1)
        y_train = train_data[target_column]
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]

        data_splits.append([X_train, X_test, y_train, y_test])

    return data_splits
