#src\kobezpipelines\prepare_data\nodes.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads raw data and performs basic cleaning."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_parquet(filepath)
    df.dropna(inplace=True)
    logger.info(f"Cleaned data size: {df.shape}")
    return df

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for machine learning."""
    df = pd.get_dummies(data)
    logger.info(f"Prepared features: {df.shape[1]}")
    return df
