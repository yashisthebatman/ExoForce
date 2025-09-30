# src/scout_hyperparameters.py
# PURPOSE: A hyper-efficient single-objective run to find promising regions.

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import warnings
import os
import joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration for SPEED ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
VALIDATION_FILENAME = 'unified_validation.csv'
SCOUT_RESULTS_FILE = 'scout_top_candidates.pkl'

N_TRIALS = 150  # We can do many trials quickly
TIMEOUT = 3600 * 1 # 1-hour timeout
N_SPLITS = 3
SUBSAMPLE_RATIO = 0.5 # Use a small sample for the scout

def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray) -> float:
    # This is a SINGLE-OBJECTIVE function, returning only the F1 score.
    max_depth = trial.suggest_int('max_depth', 5, 12)
    params = {
        'objective': 'multiclass', 'num_class': len(np.unique(y)), 'metric': 'multi_logloss',
        'seed': 42, 'n_jobs': -1, 'verbose': -1, 'class_weight': 'balanced',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 400, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': max_depth, 'num_leaves': trial.suggest_int('num_leaves', 31, min(2**max_depth - 1, 255)),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 25.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 25.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
    }
    if params['boosting_type'] == 'dart':
        params.pop('min_child_weight', None)
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.5, 1.0)
    else:
        params['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True)

    model = lgb.LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scores = []
    
    for step, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
        
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average='weighted')
        scores.append(score)
        
        trial.report(score, step)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return np.mean(scores)

def main():
    print("--- Starting ExoForge Scouting Run ---")
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    validation_df = pd.read_csv(os.path.join(DATA_DIR, VALIDATION_FILENAME))
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    X = combined_df.drop(columns=[TARGET_COLUMN])
    y_raw = combined_df[TARGET_COLUMN]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X, _, y, _ = train_test_split(X, y, train_size=SUBSAMPLE_RATIO, shuffle=True, stratify=y, random_state=42)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True)

    print(f"Scout run complete. Best F1 score: {study.best_value:.5f}")
    
    best_trials = sorted(study.best_trials, key=lambda t: t.value, reverse=True)[:5]
    top_params = [t.params for t in best_trials]

    save_path = os.path.join(ARTIFACT_DIR, SCOUT_RESULTS_FILE)
    joblib.dump(top_params, save_path)
    
    print(f"Top {len(top_params)} parameter sets from scout run saved to '{save_path}'.")
    print("\n--- Next Step: Run 'train_optimizer.py' which will now be warm-started with these results. ---")

if __name__ == '__main__':
    main()