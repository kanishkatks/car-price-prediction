# Car Price Prediction Project

This project demonstrates a complete machine learning workflow for predicting car prices based on various features. It covers data exploration, cleaning, preprocessing, feature engineering, model training using multiple regression algorithms, and performance comparison.

## Goals

*   Explore and understand the car dataset (`cars_dataset.csv`), identifying key characteristics, distributions, and potential data quality issues.
*   Clean the data by handling anomalies and ensuring consistency.
*   Develop a robust preprocessing pipeline using Scikit-learn to handle numerical (imputation, scaling) and categorical (imputation, encoding) features.
*   Perform feature engineering to potentially create more informative features for the models.
*   Train several different regression models (e.g., Linear Regression, Random Forest, Gradient Boosting, etc.) on the preprocessed data.
*   Evaluate and compare the performance of these models using appropriate metrics (like R²) on a held-out test set.
*   Identify the best-performing model based on the evaluation results.


## Dataset

The dataset (`cars_dataset.csv`) contains various specifications for 205 car models. Key features include physical attributes (body style, dimensions, weight), engine/fuel specs (type, size, horsepower, fuel system), efficiency metrics (MPG), and the target variable, `price`.

## Methods & Workflow

1.  **Exploratory Data Analysis (EDA):**
    *   Loaded and inspected the data (shape, types, basic stats).
    *   Analyzed the target variable (`price`) distribution and skewness.
    *   Visualized relationships between features and the target variable.
    *   Checked for correlations between numerical features.
2.  **Data Cleaning:**
    *   Mapped text-based numerical representations (e.g., `cylindernumber`, `doornumber`) to actual numbers.
    *   Checked and handled duplicates and missing values (if any were found after initial checks).
3.  **Preprocessing Pipeline (Scikit-learn):**
    *   Separated features (X) and target (y).
    *   Split data into Training and Testing sets.
    *   **Numerical Features:** Applied imputation (e.g., median) and scaling (e.g., `RobustScaler` or `StandardScaler`).
    *   **Categorical Features:** Applied imputation (e.g., most frequent) and encoding (e.g., `OneHotEncoder`).
    *   Combined these steps using `ColumnTransformer` to create a unified preprocessor applied to the training data (`fit_transform`) and test data (`transform`).
4.  **Model Training:**
    *   Instantiated multiple regression models (e.g., Linear Regression, Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor).
    *   Trained each model on the preprocessed training data.
5.  **Model Evaluation & Comparison:**
    *   Made predictions using each trained model on the preprocessed test data.
    *   Calculated performance metrics (R², MAE, RMSE) for each model on the test set predictions.
    *   Compared the scores to identify the best-performing model(s) for this specific task and dataset split.

## Results

*   The EDA provided valuable insights into the dataset's structure and feature relationships.
*   A preprocessing pipeline was successfully implemented to prepare the data for modeling.
*   Multiple regression models were trained and evaluated.
*   Model **Gradient BoostRegressor]** achieved the best performance on the test set with an R² score of **0.91** Other models like **Random Forest Regressor  and AdaBoost Regressor** achieved scores of **0.89**.


## Future Improvements

*   **Advanced Feature Engineering:** Explore more complex interactions or polynomial features.
*   **Target Transformation:** Apply transformations (like log transform) to the `price` if skewness negatively impacts model performance (especially linear models).
*   **More Models:** Experiment with other algorithms like XGBoost, LightGBM, CatBoost, SVR, or Neural Networks.
*   **Cross-Validation:** Implement cross-validation during training and tuning for more robust performance estimates.
*   **Ensemble Methods:** Combine predictions from multiple models.
