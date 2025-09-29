# ExoForge/src/train_optimizer.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.visualization import plot_pareto_front
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import warnings
import os
import joblib
import json
import ast

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

# --- Configuration: Faster & More Efficient Search ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
VALIDATION_FILENAME = 'unified_validation.csv'

# --- EFFICIENCY UPGRADE ---
N_SPLITS = 3  # Reduced from 5 for a ~40% speedup per trial
N_TRIALS = 300 # Reduced from 1000 for a focused, faster search
TIMEOUT = 3600 * 3 # 3-hour timeout for a more practical runtime

def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float, float]:
    """
    Three-Objective function for finding a performant, robust, and simple model.
    """
    max_depth = trial.suggest_int('max_depth', 5, 15)
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss',
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'class_weight': 'balanced',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 400, 3000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': max_depth,
        'num_leaves': trial.suggest_int('num_leaves', 31, 2**max_depth - 1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 25.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 25.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
    }

    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.5, 1.0)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_f1_scores_per_class, fold_f1_scores_weighted, fold_complexities = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        val_preds = model.predict(X_val)
        per_class_f1 = f1_score(y_val, val_preds, average=None, labels=np.unique(y))
        weighted_f1 = f1_score(y_val, val_preds, average='weighted')
        complexity = model.best_iteration_ if model.best_iteration_ > 0 else params['n_estimators']

        fold_f1_scores_per_class.append(per_class_f1)
        fold_f1_scores_weighted.append(weighted_f1)
        fold_complexities.append(complexity)

    mean_weighted_f1 = np.mean(fold_f1_scores_weighted)
    min_per_class_f1 = np.min(np.mean(fold_f1_scores_per_class, axis=0))
    mean_complexity = np.mean(fold_complexities)
    
    return mean_weighted_f1, min_per_class_f1, mean_complexity

def run_optimizer():
    print("--- Starting ExoForge Grand Unified Pipeline (Fast Mode) ---")
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    validation_df = pd.read_csv(os.path.join(DATA_DIR, VALIDATION_FILENAME))
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    X = combined_df.drop(columns=[TARGET_COLUMN])
    y_raw = combined_df[TARGET_COLUMN]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=40, multivariate=True)
    
    study = optuna.create_study(sampler=sampler, 
                                directions=['maximize', 'maximize', 'minimize'], 
                                study_name='ExoForge_Fast_Search')

    study.optimize(lambda trial: objective(trial, X, y), 
                   n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True)

    print("\n--- Discovery Phase Finished ---")
    pareto_front_trials = study.best_trials
    
    results = []
    for trial in pareto_front_trials:
        results.append({
            "Trial": trial.number,
            "Overall F1": trial.values[0],
            "Weakest Class F1": trial.values[1],
            "Complexity (Trees)": trial.values[2],
            "Params": str(trial.params)
        })
    results_df = pd.DataFrame(results).sort_values(by="Overall F1", ascending=False).reset_index(drop=True)
    
    print("\n--- Table of Optimal Models (Pareto Front) ---")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 120):
        print(results_df[['Trial', 'Overall F1', 'Weakest Class F1', 'Complexity (Trees)']])
    
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(ARTIFACT_DIR, 'pareto_front_results.csv'), index=False)
    
    best_overall_trial_params = ast.literal_eval(results_df.loc[0, 'Params'])
    param_path = os.path.join(ARTIFACT_DIR, 'best_params_from_pareto.json')
    with open(param_path, 'w') as f:
        json.dump(best_overall_trial_params, f, indent=4)
    print(f"\nParameters for the highest-performing model saved to: {param_path}")

    fig = plot_pareto_front(study, target_names=["Overall F1", "Weakest Class F1", "Complexity (Trees)"])
    plot_path = os.path.join(ARTIFACT_DIR, 'pareto_front_final.html')
    fig.write_html(plot_path)
    print(f"Interactive Pareto front plot and full results saved to '{ARTIFACT_DIR}' directory.")
    print("\n--- Next Step: Run 'finalize_and_report.py' to build and analyze the champion ensemble. ---")

if __name__ == '__main__':
    run_optimizer()