# src/run_sota_search.py
# ULTIMATE PURIST VERSION - Uses "Source-Based" GroupKFold for maximum robustness.

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import warnings
import os
import json

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
N_SPLITS = 3 # Use 3 splits, as we have 3 data sources (koi, toi, k2)

QUALIFIER_STUDY_NAME = "ExoForge_Ultimate_Qualifier"
QUALIFIER_STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/ultimate_qualifier.db"
N_QUALIFIER_TRIALS = 600
N_FINALISTS_TO_PROMOTE = 40
FINALS_STUDY_NAME = "ExoForge_Ultimate_Finals"
FINALS_STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/ultimate_finals.db"
LEADERBOARD_FILE = 'sota_pareto_front_specialists.csv'

def objective_qualifier(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray, kf: GroupKFold, groups: np.ndarray) -> float:
    params = {
        'objective': 'multiclass', 'metric': 'multi_logloss', 'seed': 42, 'n_jobs': -1,
        'class_weight': 'balanced', 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300), 'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100), 'verbosity': -1
    }
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.5, 1.0)
    else:
        params['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True)
    
    scores = []
    for train_idx, val_idx in kf.split(X, y, groups=groups):
        X_train, y_train, X_val, y_val = X.iloc[train_idx], y[train_idx], X.iloc[val_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        scores.append(f1_score(y_val, model.predict(X_val), average='weighted'))
    avg_score = np.mean(scores); trial.report(avg_score, 1); return avg_score

def objective_finals_manual(params: dict, X: pd.DataFrame, y: np.ndarray, kf: GroupKFold, groups: np.ndarray) -> tuple[float, float, float]:
    run_params = params.copy(); run_params['verbosity'] = -1
    fold_per_class_f1 = []
    for train_idx, val_idx in kf.split(X, y, groups=groups):
        X_train, y_train, X_val, y_val = X.iloc[train_idx], y[train_idx], X.iloc[val_idx], y[val_idx]
        model = lgb.LGBMClassifier(**run_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        fold_per_class_f1.append(f1_score(y_val, model.predict(X_val), average=None, labels=range(len(np.unique(y)))))
    return np.mean(fold_per_class_f1, axis=0)

def main():
    print("--- Starting ULTIMATE SOTA Search (Source-Based Adversarial CV) ---")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    
    if 'source' not in train_df.columns:
        raise ValueError("'source' column not found. Please run the latest create_dataset.py script.")
    
    groups, X = train_df['source'], train_df.drop(columns=[TARGET_COLUMN, 'source'])
    y, kf = LabelEncoder().fit_transform(train_df[TARGET_COLUMN]), GroupKFold(n_splits=N_SPLITS)
    
    qualifier_study = optuna.create_study(study_name=QUALIFIER_STUDY_NAME, storage=QUALIFIER_STORAGE_PATH, load_if_exists=True, direction='maximize', pruner=optuna.pruners.MedianPruner())
    qualifier_study.optimize(lambda trial: objective_qualifier(trial, X, y, kf, groups=groups), n_trials=N_QUALIFIER_TRIALS, show_progress_bar=True)
    
    completed_trials = sorted(qualifier_study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]), key=lambda t: t.value, reverse=True)
    finalist_trials = completed_trials[:N_FINALISTS_TO_PROMOTE]
    
    finals_study = optuna.create_study(study_name=FINALS_STUDY_NAME, storage=FINALS_STORAGE_PATH, load_if_exists=True, directions=['maximize', 'maximize', 'maximize'])
    for finalist in finalist_trials:
        if any(past_trial.params == finalist.params for past_trial in finals_study.trials): continue
        trial_values = objective_finals_manual(finalist.params, X, y, kf, groups=groups)
        finals_study.add_trial(optuna.trial.create_trial(params=finalist.params, distributions=finalist.distributions, values=list(trial_values)))

    pareto_front = finals_study.best_trials
    results_df = pd.DataFrame([{ "Trial": t.number, "f1_candidate": t.values[0], "f1_confirmed": t.values[1], "f1_false_positive": t.values[2], "Params": json.dumps(t.params) } for t in pareto_front if t.state == TrialState.COMPLETE])
    print("\nFinal Pareto Front:"); print(results_df)
    results_df.to_csv(os.path.join(ARTIFACT_DIR, LEADERBOARD_FILE), index=False)
    print(f"\nTournament winners saved to '{LEADERBOARD_FILE}'.")

if __name__ == '__main__':
    main()