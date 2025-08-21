from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def build_ffn(hidden=(128,64), random_state=42):
    return MLPClassifier(hidden_layer_sizes=hidden, activation='relu', max_iter=300, random_state=random_state)

def build_rf(random_state=42):
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=random_state
    )

def build_xgb(scale_pos_weight=1, random_state=42):
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric='logloss', use_label_encoder=False,
        scale_pos_weight=scale_pos_weight, random_state=random_state, n_jobs=-1
    )
