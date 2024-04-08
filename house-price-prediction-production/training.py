"""Processors for the model training step of the worklow."""

import logging
import os.path as op

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH,
)
from ta_lib.regression.api import SKLStatsmodelOLS


logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    transformed_columns = load_pipeline(
        op.join(artifacts_folder, "transformed_columns.joblib")
    )
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # sample data if needed. Useful for debugging/profiling purposes.
    sample_frac = params.get("sampling_fraction", None)
    if sample_frac is not None:
        logger.warn(f"The data has been sample by fraction: {sample_frac}")
        sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
    else:
        sample_X = train_X
    sample_y = train_y.loc[sample_X.index]
    train_y = train_y.values.ravel()
    # transform the training data
    train_X = features_transformer.fit_transform(train_X)
    train_X = get_dataframe(
        train_X,
        transformed_columns,
    )

    # create training pipeline
    lin_reg = Pipeline([("Linear Regression", SKLStatsmodelOLS())])

    # fit the training pipeline
    lin_reg.fit(train_X, train_y)

    # save fitted training pipeline
    save_pipeline(
        lin_reg, op.abspath(op.join(artifacts_folder, "lin_reg_train_pipeline.joblib"))
    )

    # create training pipeline
    tree_reg = Pipeline(
        [("Decision Tree", DecisionTreeRegressor(random_state=context.random_seed))]
    )

    # fit the training pipeline
    tree_reg.fit(train_X, train_y)

    # save fitted training pipeline
    save_pipeline(
        tree_reg,
        op.abspath(op.join(artifacts_folder, "tree_reg_train_pipeline.joblib")),
    )

    # Using best params
    parameters = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]
    est = RandomForestRegressor(random_state=context.random_seed)
    grid_search = GridSearchCV(
        est, parameters, cv=5, scoring="neg_mean_squared_error", return_train_score=True
    )

    grid_search.fit(train_X, train_y)

    # create training pipeline
    forest_reg = Pipeline([("Random Forest", grid_search.best_estimator_)])

    # fit the training pipeline
    forest_reg.fit(train_X, train_y)

    # save fitted training pipeline
    save_pipeline(
        forest_reg,
        op.abspath(op.join(artifacts_folder, "forest_reg_train_pipeline.joblib")),
    )

    # create training pipeline
    svm_reg = Pipeline([("SVR", SVR(kernel="linear"))])

    # fit the training pipeline
    svm_reg.fit(train_X, train_y)

    # save fitted training pipeline
    save_pipeline(
        svm_reg, op.abspath(op.join(artifacts_folder, "svm_reg_train_pipeline.joblib"))
    )
