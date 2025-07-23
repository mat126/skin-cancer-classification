# 🧪 Trained Models (`models/`)

This folder stores the serialized versions (`.pkl`) of the trained models developed during the project  
*“Hybrid CNN Architecture for Skin Cancer Classification”* (work in progress).

These models are saved using `pickle` after hyperparameter tuning and training, and are intended for later use in evaluation or deployment.

---

## 📂 Contents

- **`random_forest_best.pkl`**  
  Trained Random Forest model with optimal hyperparameters selected via `GridSearchCV`.

- **`xgboost_model.pkl`**  
  Trained XGBoost model including best configuration, incorporating class imbalance handling with `scale_pos_weight`.

---

## ⚠️ Notes

- To load the models in Python:

```python
import pickle

with open("models/random_forest_best.pkl", "rb") as f:
    model_rf = pickle.load(f)

with open("models/xgboost_model.pkl", "rb") as f:
    model_xgb = pickle.load(f)
```

---

## 🚧 Work in Progress

More models (CNNs, ensemble stacks) will be added here as the project develops.