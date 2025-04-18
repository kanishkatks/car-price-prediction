# Car Price Prediction: Data Preprocessing Pipeline

## Overview

This project walks through the creation of a data preprocessing pipeline designed to prepare car data for price prediction using machine learning. The focus is on cleaning the data, exploring it to understand its characteristics, and building a reusable pipeline with Scikit-learn.

## Goals

*   To clean and prepare a dataset containing car specifications for machine learning.
*   To perform Exploratory Data Analysis (EDA) to understand the data, especially the target variable (`price`).
*   To construct an efficient and robust Scikit-learn preprocessing pipeline that can handle both numerical and categorical features.
*   To create a pipeline that can be easily integrated with various machine learning models for predicting car prices.

## Dataset

*   **Source:** The data was sourced from Wagon Public Datasets on AWS S3.
*   **Content:** The dataset includes details for 205 different car models across 24 features (after dropping the ID column).
*   **Target Variable:** `price` (the car's price).
*   **Features:** A mix of numerical and categorical features, such as `wheelbase`, `horsepower`, `enginesize`, `fueltype`, `carbody`, `drivewheel`, etc.

## Methods & Workflow

The process followed in the notebook includes:

1.  **Data Loading & Initial Cleaning:**
    *   Fetched the dataset and loaded it using pandas.
    *   Removed the irrelevant `car_ID`.
    *   Converted text representations of numbers (like `cylindernumber`) into actual numerical values.
    *   Checked for and confirmed no duplicate rows or missing values.
2.  **Exploratory Data Analysis (EDA):**
    *   Examined the structure and types of data.
    *   Visualized the distribution of the `price` variable using histograms, box plots, and Q-Q plots to check for skewness and outliers.
3.  **Pipeline Construction (using Scikit-learn):**
    *   Separated the features (input variables) from the target (`price`).
    *   Dropped the `CarName` feature due to its high number of unique values, which is often less useful for general prediction models.
    *   Created distinct preprocessing steps for numerical and categorical features.
    *   Combined these steps into a single `ColumnTransformer` pipeline.

## Results: The Preprocessing Pipeline

The main outcome of this notebook is a Scikit-learn `ColumnTransformer` pipeline (`preprocessor`) that applies the following steps:

1.  **For Numerical Features:**
    *   **Imputation:** Fills any missing numerical values using the median (making the pipeline robust even though the initial dataset had no missing values).
    *   **Scaling:** Scales the features using `RobustScaler`, which is less sensitive to outliers.
2.  **For Categorical Features:**
    *   **Imputation:** Fills any missing categorical values using the most frequent value.
    *   **Encoding:** Converts categorical text data into a numerical format using `OneHotEncoder`. It's set to ignore unknown categories encountered during prediction.

This pipeline takes the raw feature data and transforms it into a format ready to be fed into a machine learning model.

## Technologies Used

*   **Data Handling:** pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Statistical Analysis:** Statsmodels (for Q-Q plots)
*   **Machine Learning Pipeline:** Scikit-learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `RobustScaler`, `OneHotEncoder`)
*   **Environment:** Jupyter Notebook

## Future Improvements & Next Steps

While this project focuses on the preprocessing pipeline, the logical next steps are:

*   **Model Training:** Feed the preprocessed data into various regression models (e.g., Linear Regression, Random Forest, Gradient Boosting).
*   **Model Evaluation:** Assess model performance using metrics like R-squared, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) on a separate test dataset.
*   **Hyperparameter Tuning:** Fine-tune the parameters of the best-performing models to potentially improve accuracy.
*   **Feature Engineering:** Create new, potentially more informative features from the existing ones.
*   **Feature Selection:** Identify and possibly remove features that don't contribute significantly to the prediction accuracy.
