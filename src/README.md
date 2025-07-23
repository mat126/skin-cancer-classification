# 🧠 Source Code (`src/`)

This folder contains modular Python scripts for the core components of the project  
*“Hybrid CNN Architecture for Skin Cancer Classification”* (work in progress).

Each script is structured to ensure modularity, reproducibility, and integration with notebook pipelines.

---

## 📂 Contents

- **`preprocessing.py`**  
  Contains a function `prepare_data(df)` that:
  - Removes unwanted (for lots of NaN or irrevelant to the analysis) features 
  - Identifies numerical and categorical variables
  - Builds a `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`
  - Returns feature matrix `X`, target vector `y`, and the preprocessing pipeline

- **`random_forest_src.py`**  
  Contains the function `train_random_forest()` that:
  - Builds a preprocessing + Random Forest pipeline
  - Applies hyperparameter tuning via `GridSearchCV`
  - Outputs predictions and prints classification metrics

- **`xgboost_src.py`**  
  Contains the function `train_xgboost()` that:
  - Uses XGBoost with optional `scale_pos_weight`
  - Performs `GridSearchCV` over key hyperparameters
  - Returns fitted model and predictions on test data

---

## 🚧 Work in Progress

Additional modules for neural networks, stacking methods, model evaluation, and SHAP explainability will be added as development progresses.
