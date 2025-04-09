#kobe\streamlit_app\app.py

import numpy as np
import warnings

# Monkey patch para lidar com depreciações do NumPy
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str

warnings.filterwarnings("ignore", category=DeprecationWarning)


import streamlit as st
from kedro.framework.startup import bootstrap_project
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession

# Initialize Kedro project (replace with your actual Kedro project name if different)
project_name = "kobe"

# Bootstrap Kedro environment
bootstrap_project(".")
configure_project(project_name)

st.set_page_config(page_title="Kobe Model Analysis", layout="wide")
st.title("🏀 Kobe Bryant Shot Prediction - Classifier Model Analysis")

# Start a Kedro session
with KedroSession.create() as session:
    context = session.load_context()
    catalog = context.catalog

    # Load classifier model from the Kedro Data Catalog
    if "classifier_model" in catalog.list():
        classifier_model = catalog.load("classifier_model")
        st.success("✅ Classifier model loaded successfully!")
        st.subheader("Model Object:")
        st.write(classifier_model)

        # Optional: Load comparison results from PyCaret if available
        if "comparison_df" in catalog.list():
            comparison_df = catalog.load("comparison_df")
            st.subheader("🔍 Model Comparison Results")
            st.dataframe(comparison_df)

        # Optional: Load metrics and show them
        if "classification_metrics" in catalog.list():
            metrics = catalog.load("classification_metrics")
            st.subheader("📊 Model Evaluation Metrics")
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.json(metrics["classification_report"])
    else:
        st.warning("⚠️ 'classifier_model' not found in the Data Catalog.")
