#src\kobe\pipelines\regression\pipeline.py

from kedro.pipeline import node, Pipeline, pipeline
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    """ Regression pipeline.
    This pipeline performs the following steps for regression tasks:
        1. Load and clean raw data for regression.
        2. Perform feature engineering (e.g., one-hot encoding).
        3. Train regression model using PyCaret and log results to MLflow.
        4. Evaluate model performance using validation metrics.
    Returns:
        Kedro Pipeline: A regression pipeline object. """
    return pipeline([
        # Step 1: Load and clean the raw Kobe data for regression tasks
        node(
            func=nodes.load_data,
            inputs="kobe_raw_data",
            outputs="kobe_cleaned_data_reg",
            name="regression_load_data_node"
        ),
        # Step 2: Apply feature engineering techniques (e.g., one-hot encoding)
        node(
            func=nodes.feature_engineering,
            inputs="kobe_cleaned_data_reg",
            outputs="regression_features",
            name="regression_feature_engineering_node"
        ),
        # Step 3: Train regression model using PyCaret and log experiment with MLflow
        node(
            func=nodes.train_regressor,
            inputs=["regression_features", "params:regression"],
            outputs="regressor_model",
            name="train_regressor_node"
        ),
        # Step 4: Evaluate model performance on a test split
        node(
            func=nodes.evaluate_regressor,
            inputs=["regressor_model", "regression_features", "params:regression"],
            outputs="regression_metrics",
            name="evaluate_regressor_node"
        ),
    ])
