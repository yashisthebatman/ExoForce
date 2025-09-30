# src/train_optimizer.py
# PURPOSE: To run a focused, unified search to find the best LightGBM specialists.

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import warnings, os, json, ast

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
STUDY_NAME = "ExoForge_LGBM_Specialist_Search" # Focused study name
STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/exoforge_lgbm_study.db"
N_TRIALS = 300 # Can be reduced slightly as the search space is simpler
N_SPLITS = 3

def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float, float]:
    # This objective is now ONLY for LightGBM
    params = {
        'objective': 'multiclass', 'metric': 'multi_logloss', 'seed': 42, 'n_jobs': -1, 'verbose': -1,
        'class_weight': 'balanced', 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    if params['boosting_type'] == 'dart':
        params.pop('min_child_weight', None)
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.5, 1.0)
    else: # GBDT
        params['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True)
    
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

def run_optimizer():
    print("--- Starting Focused LightGBM Specialist Search ---")
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    X = train_df.drop(columns=[TARGET_COLUMN])
    y = LabelEncoder().fit_transform(train_df[TARGET_COLUMN])

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        directions=['maximize', 'maximize', 'maximize'], 
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True
    )
    
    print(f"Study '{STUDY_NAME}' loaded/created with {len(study.trials)} trials.")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS, show_progress_bar=True)
    
    print(f"\n--- Search Finished. Total trials in study: {len(study.trials)} ---")
    pareto_front_trials = study.best_trials
    
    results = []
    for trial in pareto_front_trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                "Trial": trial.number,
                "f1_candidate": trial.values[0],
                "f1_confirmed": trial.values[1],
                "f1_false_positive": trial.values[2],
                "Params": str(trial.params)
            })
    
    results_df = pd.DataFrame(results)
    print("\n--- Pareto Front of LightGBM Specialist Models ---")
    print(results_df.head())
    
    results_df.to_csv(os.path.join(ARTIFACT_DIR, 'lgbm_specialist_pareto.csv'), index=False)
    print(f"\nLGBM specialist candidates saved to 'lgbm_specialist_pareto.csv'.")
    print("\n--- Next Step: Run 'train_base_models.py' to build the diverse committee. ---")

if __name__ == '__main__':
    run_optimizer()