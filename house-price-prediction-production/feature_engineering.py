"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""

import logging
import os.path as op

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scripts import CombinedAttributesAdder

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)

from ta_lib.data_processing.api import Outlier

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    cat_columns = ["ocean_proximity"]
    num_columns = train_X.columns.drop("ocean_proximity")

    # NOTE: You can use ``Pipeline`` to compose a collection of transformers
    # into a single transformer. In this case, we are composing a

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    # NOTE: the list of transformations here are not sequential but weighted
    # (if multiple transforms are specified for a particular column)
    # for sequential transforms use a pipeline as shown above.
    features_transformer = ColumnTransformer(
        [
            ("cat", OneHotEncoder(), cat_columns),
            ("num", num_pipeline, num_columns),
        ]
    )

    # Check if the data should be sampled. This could be useful to quickly run
    # the pipeline for testing/debugging purposes (undersample)
    # or profiling purposes (oversample).
    # The below is an example how the sampling can be done on the train data if required.
    # Model Training in this reference code has been done on complete train data itself.

    # Train the feature engg. pipeline prepared earlier. Note that the pipeline is
    # fitted on only the **training data** and not the full dataset.
    # This avoids leaking information about the test dataset when training the model.
    # In the below code train_X, train_y in the fit_transform can be replaced with
    # sample_X and sample_y if required.
    train_X = features_transformer.fit_transform(train_X)

    # Note: we can create a transformer/feature selector that simply drops
    # a specified set of columns. But, we don't do that here to illustrate
    # what to do when transformations don't cleanly fall into the sklearn
    # pattern.
    transformed_columns = (
        features_transformer.named_transformers_["cat"]
        .get_feature_names_out(input_features=cat_columns)
        .tolist()
    )
    transformed_columns += num_columns.tolist() + [
        "rooms_per_household",
        "population_per_household",
        "bedrooms_per_room",
    ]

    train_X = get_dataframe(
        train_X,
        transformed_columns,
    )

    # saving the list of relevant columns and the pipeline.
    save_pipeline(
        transformed_columns,
        op.abspath(op.join(artifacts_folder, "transformed_columns.joblib")),
    )
    save_pipeline(
        features_transformer, op.abspath(op.join(artifacts_folder, "features.joblib"))
    )
