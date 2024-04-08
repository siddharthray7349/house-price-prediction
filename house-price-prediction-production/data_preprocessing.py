"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
)


@register_processor("data-cleaning", "housing")
def clean_order_table(context, params):
    """Clean the ``HOUSING`` data table.

    The table containts the HOUSING data and has information on the median house price,
    the bedrooms, the population etc.
    """

    input_dataset = "raw/housing"
    output_dataset = "processed/housing"

    # load dataset
    housing_df = load_dataset(context, input_dataset)

    housing_df_processed = (
        housing_df
        # tweak to test pipeline quickly or profile performance
        # .sample(frac=1, replace=False)
        # any additional processing/cleaning
    )

    # save dataset
    save_dataset(context, housing_df_processed, output_dataset)
    return housing_df_processed


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``SALES`` table into ``train`` and ``test`` datasets."""

    input_dataset = "processed/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df_processed = load_dataset(context, input_dataset)

    # creating additional features that are not affected by train test split. These are features that are processed globally
    housing_df_processed["income_cat"] = pd.cut(
        housing_df_processed["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_processed, splitter, by="income_cat"
    )

    for set_ in (housing_df_train, housing_df_test):
        set_.drop("income_cat", axis=1, inplace=True)

    target_col = params["target"]
    # split train dataset into features and target
    train_X, train_y = housing_df_train.get_features_targets(
        target_column_names=target_col
    )
    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = housing_df_test.get_features_targets(
        target_column_names=target_col
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
