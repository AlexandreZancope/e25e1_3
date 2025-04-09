#src\kobe\pipelines\prepare_data\pipeline.py

from kedro.pipeline import node, Pipeline
from .nodes import load_and_clean_data, prepare_features

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_and_clean_data,
            inputs="kobe_raw_data",
            outputs="cleaned_data",
            name="load_and_clean_data_node"
        ),
        node(
            func=prepare_features,
            inputs="cleaned_data",
            outputs="prepared_features",
            name="prepare_features_node"
        )
    ])