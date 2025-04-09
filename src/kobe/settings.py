#src\kobe\settings.py

# src/kobe/settings.py

from kedro.config import OmegaConfigLoader
from kedro_mlflow.framework.hooks import MlflowHook
from kedro.pipeline import Pipeline
from kobe.pipelines.classification import create_pipeline as classification_pipeline

HOOKS = (MlflowHook(),)

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": classification_pipeline(),
        "classification": classification_pipeline(),
    }
