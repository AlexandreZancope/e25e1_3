# src\kobe\pipeline_registry.py

"""Project pipelines."""

from kedro.pipeline import Pipeline
from kobe.pipelines.prepare_data import pipeline as prepare_data_pipeline
from kobe.pipelines.classification import create_pipeline as classification_pipeline
from kobe.pipelines.regression import create_pipeline as regression_pipeline
from typing import Dict


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    - Classification Pipeline: Handles predicting whether Kobe scored a basket.
    - Regression Pipeline: Predicts the probability of scoring.
    - Default: Runs all pipelines for comprehensive analysis.
    
    Returns:
        dict: A dictionary with pipeline names as keys and Kedro pipeline objects as values.
    """
    return {
        "prepare_data": prepare_data_pipeline.create_pipeline(),
        "classification": classification_pipeline(),
        "regression": regression_pipeline(),
        "__default__": prepare_data_pipeline.create_pipeline() + classification_pipeline() + regression_pipeline(),
    }
