from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train_random_forest(X_train, y_train, X_test, y_test, preprocessor):
    pipeline_rf = Pipeline([
        ('preprocessing', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid_rf = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__class_weight': ['balanced']
    }

    grid_rf = GridSearchCV(
        pipeline_rf,
        param_grid_rf,
        cv=3,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )

    grid_rf.fit(X_train, y_train)
    y_pred_rf = grid_rf.predict(X_test)
    
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, digits=4))

    return grid_rf, y_pred_rf
