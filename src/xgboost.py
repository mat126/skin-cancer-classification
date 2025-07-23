from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train_xgboost(X_train, y_train, X_test, y_test, preprocessor, scale_pos_weight=1):
    """
    Trains an XGBoost classifier with hyperparameter tuning via GridSearchCV.
    
    Parameters:
        X_train, y_train: training data
        X_test, y_test: test data
        preprocessor: a ColumnTransformer for preprocessing
        scale_pos_weight: class imbalance weight
    
    Returns:
        grid_xgb: trained GridSearchCV object
        y_pred_xgb: predicted labels for X_test
    """
    pipeline_xgb = Pipeline([
        ('preprocessing', preprocessor),
        ('xgb', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ))
    ])

    param_grid_xgb = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.05, 0.1],
        'xgb__scale_pos_weight': [scale_pos_weight]
    }

    grid_xgb = GridSearchCV(
        pipeline_xgb,
        param_grid_xgb,
        cv=3,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )

    grid_xgb.fit(X_train, y_train)
    y_pred_xgb = grid_xgb.predict(X_test)

    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb, digits=4))

    return grid_xgb, y_pred_xgb
