#src\kobe\pipelines\regression\nodes.py

import logging
import pandas as pd
from typing import Dict, Any
from pycaret.regression import setup, compare_models, pull
import mlflow
import numpy as np

logger = logging.getLogger(__name__)

def load_data(data: pd.DataFrame) -> pd.DataFrame:
    """Loads and cleans raw data."""
    logger.info(f"Initial dataset size: {data.shape}")
    df = data.copy()
    df.dropna(inplace=True)
    logger.info(f"Dataset size after dropping missing values: {df.shape}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering for regression."""
    result = pd.get_dummies(df)
    logger.info(f"Number of features after encoding: {result.shape[1]}")
    logger.info(f"Generated feature columns: {list(result.columns)}")
    return result

def train_regressor(data: pd.DataFrame, params: Dict[str, Any]) -> Any:
    """Trains a regression model using PyCaret and logs in MLflow."""
    mlflow_tracking_uri = params.get("mlflow_tracking_uri")
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = params.get("experiment_name", "default_regression_experiment")
    mlflow.set_experiment(experiment_name)

    logger.info("Setting up PyCaret regression environment...")

    if params["target"] not in data.columns:
        raise ValueError(f"Target column '{params['target']}' not found in data.")

    setup(
        data=data,
        target=params["target"],
        session_id=params.get("session_id", 123),
        train_size=1 - params.get("test_size", 0.2),
        silent=True,
        log_experiment=True,
        experiment_name=experiment_name,
        log_plots=True
    )

    logger.info("Comparing regression models...")
    best_model = compare_models()
    comparison_df = pull()

    comparison_df.to_csv("comparison_results.csv", index=False)
    mlflow.log_artifact("comparison_results.csv")

    logger.info("Best regression model trained successfully.")
    return best_model, comparison_df

def evaluate_regressor(model: Any, features: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the regression model using RMSE and R²."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    logger.info("Evaluating regression model...")

    X = features.drop(params["target"], axis=1)
    y = features[params["target"]]

    test_ratio = params.get("test_size", 0.2)
    _, X_test, _, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R²: {r2:.4f}")

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    return {
        "rmse": rmse,
        "r2_score": r2
    }