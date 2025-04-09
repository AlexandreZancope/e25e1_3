#src\kobe\pipelines\classification\pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes
from pprint import pprint


def log_data_summary(data):
    """
    Logs basic statistics of the raw dataset for inspection.

    Args:
        data: Raw input DataFrame.

    Returns:
        Same input DataFrame for chaining.
    """
    print("📊 Dataset Summary:")
    pprint(data.describe(include="all"))
    return data


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the classification pipeline with the following steps:
        1. Log a summary of the raw data (debug).
        2. Clean the raw data (remove missing values).
        3. Feature engineering (one-hot encoding).
        4. Train classifier using PyCaret.
        5. Evaluate model and log results to MLflow.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline([
        # Step 1: Log summary of raw data
        node(
            func=log_data_summary,
            inputs="kobe_raw_data",
            outputs="kobe_raw_data_logged",
            name="log_data_summary_node"
        ),
        # Step 2: Load and clean raw data
        node(
            func=nodes.load_data,
            inputs="kobe_raw_data_logged",
            outputs="kobe_cleaned_data",
            name="load_data_node"
        ),
        # Step 3: Feature engineering (e.g., one-hot encoding)
        node(
            func=nodes.feature_engineering,
            inputs="kobe_cleaned_data",
            outputs="classification_features",
            name="feature_engineering_node"
        ),
        # Step 4: Train classifier with PyCaret + MLflow
        node(
            func=nodes.train_classifier,
            inputs=["classification_features", "params:classification"],
            outputs=["classifier_model", "comparison_df"],
            name="train_classifier_node"
        ),
        # Step 5: Evaluate model and log metrics to MLflow
        node(
            func=nodes.evaluate_classifier,
            inputs=["classifier_model", "classification_features", "params:classification.metrics"],
            outputs="classification_metrics",
            name="evaluate_classifier_node"
        ),
    ])
