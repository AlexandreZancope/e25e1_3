KOBE PROJECT

### OVERVIEW
    This project uses a Machine Learning workflow to predict hit/miss and hit probability in shots using PyCaret and MLflow, integrating with a Streamlit visualization interface.
    This project aims to develop a shot predictor using two approaches (regression and classification) to anticipate whether "Black Mamba" (Kobe's nickname) made or missed a shot.

### STRUCTURE
    . data/01_raw: Raw input data.
        - dataset_kobe_dev.parquet
        - dataset_kobe_prod.parquet
    . data/02_intermediate: Processed data for intermediate use.
    . data/04_feature: Data with extracted features for modeling.
    . data/06_models: Trained models.
    . data/08_reporting: Final reports and results.

### WORKFLOW
1. Data Preparation:
    . Begin with kobe_dataset in 01_raw.
    . Prepare data in 02_intermediate.
    ## Raw data is processed in 02_intermediate for cleaning and organization.

2. Feature Engineering:
    Extraction of important features in 04_feature.

3. Modeling:
    . Classification: Predict hit/miss using PyCaret/Sklearn.
    . Regression: Predict probability of a successful shot.
    ## Training using PyCaret/SKLearn.

4. Training:
    . Execute model training for both approaches.

5. Validation:
    . Evaluate models with metrics and visualizations.

6. Model Registration:
    . Use MLflow for model tracking.

7. Inference via Streamlit:
    . Provide an interface for predictions.

8. Reporting:
    . Results and outputs are saved in 08_reporting.

### ITEN 2
### DIAGRAM
### The Diagram asked on the Question 2 is located at:
    https://github.com/AlexandreZancope/25E1_3-ml-projetofinal-20250405/blob/main/pd-diagrama-1.png

### HOW TO RUN
1. Environment Setup:
    . Install dependencies from requirements_311.txt.

2. Pipeline Execution:
    . Scripts for execution are in src/kobe/pipelines.

3. Streamlit Interface:
    . Run streamlit run streamlit_app/app.py.


### The entire structure is saved at:


### ITEN 3
### How the tools Streamlit, MLFlow, Pycaret and Scikit-Learn can help on building the described pipelines above? The answer must be performed on the following aspects:
The tools Streamlit, MLflow, PyCaret, and Scikit-Learn are essential for different stages of developing machine learning models, each offering distinct functionalities that complement the process. Here's how each assists in the requested aspects:
### 1. Experiment Tracking
    MLflow: Essential for managing and tracking ML experiments. It allows logging metrics, parameters, and artifacts from each run, making it easy to compare different models and configurations.
    PyCaret: Integrates directly with MLflow to facilitate experiment tracking without heavy manual configuration.

### 2. Training Functions
    Scikit-Learn: Provides a wide range of supervised and unsupervised learning algorithms and data preprocessing tools, crucial for model training.
    PyCaret: Abstracts the complexity of Scikit-Learn, allowing for quick training and tuning of models with just a few lines of code, optimizing experiments efficiently.

### 3. Model Health Monitoring
    MLflow: Offers monitoring functionalities through continuous logging of metrics and results, helping identify any model performance degradation over time.
    Streamlit: Can be used to create interactive dashboards that showcase model performance over time, providing continuous visualization of model health metrics.

### 4. Model Updating
    PyCaret: Facilitates model updates through tuning and retraining techniques with cross-validation, making the update process more efficient.
    Scikit-Learn: Provides pipelines that simplify the reintegration of new data for quick updates.

### 5. Provisioning (Deployment)
    Streamlit: Enables the creation of user-friendly, interactive web applications to display model outputs in a friendly format.
    MLflow: With its "MLflow Models" module, it allows distribution and deployment of models in production services with support for various deployment tools.

These tools, when used together, provide a robust infrastructure for the development, monitoring, and implementation of machine learning models, facilitating the entire project lifecycle.


### ITEN 4
### Based on the created diagram into question (iten 2), point out the artifacts that will be created during a project life. For each artifacts, the detailed description from its composition.
Based on the diagram provided, here are the artifacts that will be created throughout the project, along with detailed descriptions of their composition:

### 1. kobe_dataset (data/01_raw/)
    Description: The raw dataset containing all initial data collected about Kobe Bryant's shots. This dataset is unprocessed and serves as the starting point for further data manipulation and analysis.

### 2. prepare_data (data/02_intermediate/)
    Description: Intermediate data resulting from preprocessing. This involves cleaning, filtering, and possibly transforming data to be suitable for feature extraction and model input.

### 3. attributes (data/04_feature/)
    Description: A dataset of extracted features used in model training. These features are derived from the intermediate data and are key attributes relevant to predicting shot outcomes.

### 4. classification
    Description: A model focused on classification tasks to predict whether a shot is successful or not. It involves feature selection and model construction tailored to classification algorithms.

### 5. regression
    Description: A model designed for regression tasks to predict the probability of a successful shot. It uses selected features and applies regression methods to estimate probabilistic outcomes.

### 6. training (PyCaret / Sklearn)
    Description: The training process for both classification and regression models using libraries like PyCaret and Scikit-Learn. This includes model fitting and parameter tuning.

### 7. model_validations
    Description: Artifacts encompassing metrics and visualizations of model performance. Validation involves evaluating model accuracy, precision, recall, and other performance indicators.

### 8. model_registry (MLflow Tracking)
    Description: A repository for model versions and associated metadata tracked by MLflow. It logs experiments, metrics, and parameters for easy retrieval and comparison.


### 9. streamlit
    Description: A Streamlit application serving as an interface for model inference. It provides an interactive way to input data and visualize the model's predictions.

### 10. outputs / reporting (data/08_reporting/)
    Description: Final outputs and reports generated from the model's predictions and analysis. It includes summaries, detailed findings, and visual representations intended for stakeholders.

Each artifact plays a crucial role in the overall project lifecycle, from data collection to model deployment and reporting.


### ITEN 5
### Implement the pipeline of data processing with MLFlow, run with the name "PreparacaoDados":
a. The data must be located at "/data/01_raw/dataset_kobe_dev.parquet" and "/data/01raw/dataset_kobe_prod.parquet
    They are there, you can see then using tree /F:
        (kobe-env311) C:\Users\Opentech_Ps\kobe>tree /F
    Listagem de caminhos de pasta
    O número de série do volume é 1685-3967
    C:.
    │   .gitignore
    │   logs.log
    │   pyproject.toml
    │   README.md
    │   requirements.txt
    │   requirements_311.txt
    │   Untitled.ipynb
    │
    │
    ├───data
    │   ├───01_raw
    │   │       .gitkeep
    │   │       dataset_kobe_dev.parquet
    │   │       dataset_kobe_prod.parquet
    │   │

b. Pay attention there are some data missing on the database! The lines with missing data must be ignored. For this exercise, online the listed columns will be considered:
    lat
    lng
    minutes remaining
    period
    playoffs
    shot_distance
        This variable will be your target, whereas 0 (zero) means Kobe missed and 1 (one) means Kobe marked. The dataset from this operation will be stored in the folder "/data/02_intermediate/data_filteded.parquet". What is the dimension from the dataset?
    Training for 80% and test for 20% 


### ITEN 6
### Implement the pipeline of training model with the MLFlow, named "Treinamento"
a. With the separated data for training, train a model with Logistic Regression from the sklearn using the pyCaret library.
b. Register the function cost "log loss" using the test base
c. With the separated data for training, train a model fo decision tree from the skleanr using the pyCaret library
d. Register the function cost "log loss" and F1_score for tree model
e. Select one of two models to be finalized and justify your choose


### ITEN 7
### Register a model of classification and become thru MLFlow (or as a local API, or embebeeded the model in the aplication). Develope a pipeline of aplication (aplicacao.py) to load the production base (/data/01_raw/dataset_kobe_prod.parquet) and apply the model. Rename the run from the MLFlow as "PipelineAplicacao" and publish the table and the results (artefactis as .parquet), log the metrics from a new log_loos and F1_score from the model.
a. Is the model adherent to this new base/ What has been changed? Justify
b. Describe how we can monitor the model health in the scene with or without the possibility of a variable answer to the operational model.
c. Describe the reactive and predictive strategies of training to the operational model.



### ITEN 8
### Implement a dashboard of the operational monitoring using Streamlit