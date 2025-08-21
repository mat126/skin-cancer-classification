import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from src.config import CFG
from src.preprocessing import prepare_data
from src.models.tabular import build_ffn, build_rf, build_xgb
from src.utils.io import save_pickle
from src.utils.seeding import set_global_seed

def run_stacking_kfold(df: pd.DataFrame, n_splits=5, random_state=42):
    X, y, preprocessor = prepare_data(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros((len(df), 3), dtype=float)  # [rf, xgb, ffn]
    base_models = {"rf": [], "xgb": [], "ffn": []}

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        # Random Forest
        rf = build_rf()
        rf_pipe = Pipeline([('prep', preprocessor), ('rf', rf)])
        rf_pipe.fit(X_tr, y_tr)
        oof_preds[va, 0] = rf_pipe.predict_proba(X_va)[:, 1]
        base_models["rf"].append(rf_pipe)

        # XGBoost
        pos = y_tr.value_counts().get(1, 1)
        neg = y_tr.value_counts().get(0, 1)
        spw = max(1.0, neg / max(1, pos))
        xgb = build_xgb(scale_pos_weight=spw)
        xgb_pipe = Pipeline([('prep', preprocessor), ('xgb', xgb)])
        xgb_pipe.fit(X_tr, y_tr)
        oof_preds[va, 1] = xgb_pipe.predict_proba(X_va)[:, 1]
        base_models["xgb"].append(xgb_pipe)

        # Feedforward NN (sklearn MLP)
        ffn = build_ffn()
        ffn_pipe = Pipeline([('prep', preprocessor), ('ffn', ffn)])
        ffn_pipe.fit(X_tr, y_tr)
        oof_preds[va, 2] = ffn_pipe.predict_proba(X_va)[:, 1]

        print(f"[Fold {fold}] Done.")

    # Meta-model on OOF
    meta = LogisticRegression(max_iter=1000)
    meta.fit(oof_preds, y.values)
    y_oof_hat = (meta.predict_proba(oof_preds)[:, 1] > 0.5).astype(int)
    print(f"OOF F1: {f1_score(y, y_oof_hat):.4f}")

    Path(CFG.models_dir).mkdir(exist_ok=True, parents=True)
    Path(CFG.outputs_dir).mkdir(exist_ok=True, parents=True)

    save_pickle(meta, CFG.models_dir / "meta_logreg.pkl")
    save_pickle(base_models, CFG.models_dir / "base_models_folds.pkl")
    pd.DataFrame(oof_preds, columns=["rf", "xgb", "ffn"]).to_csv(
        CFG.outputs_dir / "oof_preds.csv", index=False
    )

def main():
    set_global_seed(CFG.seed)
    df = pd.read_csv(CFG.train_csv)
    run_stacking_kfold(df, n_splits=5, random_state=CFG.seed)

if __name__ == "__main__":
    main()
