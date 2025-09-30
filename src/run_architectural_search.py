# src/run_architectural_search.py
# PURPOSE: Stage A - To find the best model architectures in a low-dimensional search.

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings, os, json

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
STUDY_NAME = "ExoForge_Architecture_Search"
STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/exoforge_architecture_study.db"
LEADERBOARD_FILE = 'architecture_leaderboard.csv'
N_TRIALS = 150 # Shorter search as the space is smaller
N_SPLITS = 3
SUBSAMPLE_RATIO = 0.5 # Use a medium subsample for this stage

def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float, float]:
    # --- This objective ONLY tunes architectural parameters ---
    params = {
        'objective': 'multiclass', 'metric': 'multi_logloss', 'seed': 42, 'n_jobs': -1, 'verbose': -1,
        'class_weight': 'balanced',
        
        # --- TUNED ARCHITECTURAL PARAMS ---
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

        # --- FIXED TRAINING PARAMS (reasonable defaults) ---
        'learning_rate': 0.05,
        'n_estimators': 1000, 
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }
    # (The rest of the objective function is the same as before)
    model = lgb.LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_per_class_f1 = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val)
        fold_per_class_f1.append(f1_score(y_val, preds, average=None, labels=range(len(np.unique(y)))))
    mean_scores = np.mean(fold_per_class_f1, axis=0)
    return mean_scores[0], mean_scores[1], mean_scores[2]

def main():
    print("--- Starting Stage A: Architectural Search ---")
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    X_full = train_df.drop(columns=[TARGET_COLUMN])
    y_full = LabelEncoder().fit_transform(train_df[TARGET_COLUMN])
    X, _, y, _ = train_df.split(X_full, y_full, train_size=SUBSAMPLE_RATIO, stratify=y_full, random_state=42)
    
    study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'], study_name=STUDY_NAME, storage=STORAGE_PATH, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, show_progress_bar=True)
    
    results = []
    for trial in study.best_trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results.append({"Trial": trial.number, "values": trial.values, "Params": trial.params})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(ARTIFACT_DIR, LEADERBOARD_FILE), index=False)
    print(f"\nArchitectural search complete. Leaderboard saved to '{LEADERBOARD_FILE}'.")
    print("\n--- Next Step: Run 'run_refinement_search.py' to tune the winning architectures. ---")

if __name__ == '__main__':
    main()