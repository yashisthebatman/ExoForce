# src/run_sota_search.py
# FINAL CORRECTED VERSION - Implements a two-phase search to correctly find specialists.

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
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
N_SPLITS = 3

# --- Phase 1: Qualifier Tournament Config ---
QUALIFIER_STUDY_NAME = "ExoForge_Qualifier_Tournament"
QUALIFIER_STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/qualifier_tournament.db"
N_QUALIFIER_TRIALS = 200 # Number of trials for the fast, single-objective search
N_FINALISTS_TO_PROMOTE = 40 # Promote the top 40 candidates to the finals

# --- Phase 2: Finals Config ---
FINALS_STUDY_NAME = "ExoForge_SOTA_Finals"
FINALS_STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/sota_finals.db"
LEADERBOARD_FILE = 'sota_pareto_front_specialists.csv'


def objective_qualifier(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray, kf: StratifiedKFold) -> float:
    """
    Phase 1 Objective: Fast, single-objective search using pruning.
    Returns the AVERAGE F1-score.
    """
    params = {
        'objective': 'multiclass', 'metric': 'multi_logloss', 'seed': 42, 'n_jobs': -1, 'verbose': -1,
        'class_weight': 'balanced', 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300), 'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.5, 1.0)
    else:
        params['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True)

    # Simple 2-fold CV for speed in the qualifier
    scores = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average='weighted'))
    
    avg_score = np.mean(scores)
    # Since this is a single-objective study, we can now use trial.report()
    trial.report(avg_score, 1) 
    return avg_score

def objective_finals(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray, kf: StratifiedKFold) -> tuple[float, float, float]:
    """
    Phase 2 Objective: Rigorous, multi-objective evaluation. NO PRUNING.
    Returns a tuple of all three F1 scores.
    """
    # The trial's parameters are pre-set via enqueue_trial
    params = trial.params
    
    fold_per_class_f1 = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = model.predict(X_val)
        fold_per_class_f1.append(f1_score(y_val, preds, average=None, labels=range(len(np.unique(y)))))

    mean_scores = np.mean(fold_per_class_f1, axis=0)
    return mean_scores[0], mean_scores[1], mean_scores[2]

def main():
    print("--- Starting Two-Phase SOTA Specialist Search ---")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    X = train_df.drop(columns=[TARGET_COLUMN])
    y = LabelEncoder().fit_transform(train_df[TARGET_COLUMN])
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # --- Phase 1: Qualifier Tournament ---
    print(f"\n--- Phase 1: Running {N_QUALIFIER_TRIALS} fast trials to find finalists... ---")
    qualifier_study = optuna.create_study(
        study_name=QUALIFIER_STUDY_NAME, storage=QUALIFIER_STORAGE_PATH, load_if_exists=True,
        direction='maximize', pruner=optuna.pruners.MedianPruner()
    )
    qualifier_study.optimize(lambda trial: objective_qualifier(trial, X, y, kf), n_trials=N_QUALIFIER_TRIALS, show_progress_bar=True)
    
    print("\nQualifier tournament complete. Selecting best candidates for the finals.")
    finalist_trials = qualifier_study.best_trials[:N_FINALISTS_TO_PROMOTE]
    if not finalist_trials:
        print("Error: No successful trials in the qualifier phase. Exiting.")
        return
    print(f"Promoting {len(finalist_trials)} finalists to the multi-objective evaluation.")

    # --- Phase 2: Multi-Objective Finals ---
    print(f"\n--- Phase 2: Running rigorous multi-objective evaluation on the {len(finalist_trials)} finalists... ---")
    finals_study = optuna.create_study(
        study_name=FINALS_STUDY_NAME, storage=FINALS_STORAGE_PATH, load_if_exists=True,
        directions=['maximize', 'maximize', 'maximize']
    )

    # Enqueue all the finalists so we only evaluate the best from Phase 1
    for trial in finalist_trials:
        finals_study.enqueue_trial(trial.params)
    
    # Run optimize only for the enqueued trials
    finals_study.optimize(lambda trial: objective_finals(trial, X, y, kf), n_trials=len(finalist_trials), show_progress_bar=True)

    # --- Save Final Results ---
    print(f"\n--- Search Complete. Final Pareto front is ready. ---")
    pareto_front = finals_study.best_trials
    results = []
    for trial in pareto_front:
        if trial.state == optuna.trial.State.COMPLETE:
            results.append({
                "Trial": trial.number, "f1_candidate": trial.values[0],
                "f1_confirmed": trial.values[1], "f1_false_positive": trial.values[2],
                "Params": json.dumps(trial.params)
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(ARTIFACT_DIR, LEADERBOARD_FILE), index=False)
    print(f"\nTournament winners saved to '{LEADERBOARD_FILE}'.")
    print("\n--- Next Step: Run 'train_base_models.py' to build the specialist committee. ---")

if __name__ == '__main__':
    main()