#src\kobe\pipelines\classification\nodes.py

import logging
import pandas as pd
from typing import Dict, Any
from pycaret.classification import setup, compare_models, pull
import mlflow

logger = logging.getLogger(__name__)


def load_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Loads and cleans raw input data by dropping missing values.

    Args:
        data: Raw input DataFrame.

    Returns:
        Cleaned DataFrame with no missing values.
    """
    logger.info(f"Initial dataset size: {data.shape}")
    df = data.copy()
    df.dropna(inplace=True)
    logger.info(f"Dataset size after dropping missing values: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic feature engineering using one-hot encoding.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Feature-engineered DataFrame with encoded categorical variables.
    """
    result = pd.get_dummies(df)
    logger.info(f"Feature set after one-hot encoding: {result.shape[1]} features.")
    logger.info(f"Generated features: {list(result.columns)}")
    return result


def train_classifier(data: pd.DataFrame, params: Dict[str, Any]) -> Any:
    """
    Trains a classification model using PyCaret and logs the experiment to MLflow.

    Args:
        data: Feature-engineered dataset.
        params: Dictionary with training parameters.

    Returns:
        Tuple with best model and comparison DataFrame.
    """
    logger.info(f"Using PyCaret to train models with target: {params['target']}")

    if params["target"] not in data.columns:
        raise ValueError(f"Target column '{params['target']}' not found in data.")

    setup(
        data=data,
        target=params["target"],
        session_id=params.get("session_id", 123),
        train_size=1 - params.get("test_size", 0.2),
        normalize=params.get("preprocessing", {}).get("normalize", False),
        transform=params.get("preprocessing", {}).get("scale", False),
        silent=True,
        log_experiment=True,
        log_plots=True
    )

    included_models = params.get("models", [])
    logger.info(f"Comparing models: {included_models}")
    best_model = compare_models(include=included_models)

    comparison_df = pull()
    comparison_path = "data/08_reporting/classification_model_comparison.csv"
    comparison_df.to_csv(comparison_path)

    mlflow.log_artifact(comparison_path)
    mlflow.pyfunc.log_model("best_model", python_model=best_model)

    return best_model, comparison_df


def evaluate_classifier(model: Any, features: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the trained classifier and logs accuracy and report to MLflow.

    Args:
        model: Trained classification model.
        features: Dataset including features and the target column.
        params: Configuration parameters including test_size.

    Returns:
        Dictionary with evaluation metrics.
    """
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    test_ratio = params.get("test_size", 0.2)

    X = features.drop("shot_made_flag", axis=1)
    y = features["shot_made_flag"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    logger.info(f"Model Accuracy: {accuracy:.4f}")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_dict(report, "classification_report.json")

    return {
        "accuracy": accuracy,
        "classification_report": report
    }
