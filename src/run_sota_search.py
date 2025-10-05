# ================================================
# FILE: src/run_sota_search.py
# ================================================
# PURPOSE: KOI-FIRST & MLFLOW IMPLEMENTATION (v1.1 - Enhanced HPO)
# This script runs the SOTA search on the feature-rich KOI-only dataset.
# It uses standard StratifiedKFold, integrates with MLflow for tracking,
# and adds tunable class weights to the Optuna search space to find the
# optimal penalty for misclassifying rare classes.

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import mlflow
import warnings
import os
import json

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'koi_train.csv' # Using the new KOI-only dataset
N_SPLITS = 5 # Using 5 splits for robust validation

# --- MLflow Configuration ---
MLFLOW_EXPERIMENT_NAME = "ExoForge-KOI-Specialist-SOTA-Search"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# --- Optuna Configuration ---
N_TRIALS = 300
LEADERBOARD_FILE = 'koi_sota_pareto_front_v2.csv' # New version for this experiment


def suggest_class_weights(trial: optuna.trial.Trial) -> (str or dict):
    """
    Suggests a class weight configuration for the trial. This allows Optuna
    to search for the best way to handle class imbalance.
    """
    weight_type = trial.suggest_categorical('weight_type', ['balanced', 'custom'])
    if weight_type == 'balanced':
        return 'balanced'
    else:
        # We are most interested in correctly identifying CONFIRMED planets (class 1).
        # Let's search for the optimal penalty for misclassifying this class.
        # Class Mappings: 0: CANDIDATE, 1: CONFIRMED, 2: FALSE POSITIVE
        confirmed_weight = trial.suggest_float('confirmed_weight', 2.0, 20.0)
        return {0: 1, 1: confirmed_weight, 2: 1}


def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray, kf: StratifiedKFold) -> tuple:
    """
    Unified Optuna objective function that is logged with MLflow.
    It returns per-class F1 scores for multi-objective optimization.
    """
    # Start a nested MLflow run for this specific Optuna trial
    with mlflow.start_run(nested=True):
        
        class_weights = suggest_class_weights(trial)

        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'seed': 42,
            'n_jobs': -1, # Use all available CPU cores
            'class_weight': class_weights, # <-- The new tunable parameter
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'verbosity': -1
        }
        if params['boosting_type'] == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.5, 1.0)

        # Log params to MLflow, ensuring the dictionary is converted to a string
        loggable_params = params.copy()
        if isinstance(loggable_params['class_weight'], dict):
            loggable_params['class_weight'] = json.dumps(loggable_params['class_weight'])
        mlflow.log_params(loggable_params)
        
        fold_per_class_f1 = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(X_val)
            fold_per_class_f1.append(f1_score(y_val, preds, average=None, labels=range(len(np.unique(y)))))
        
        mean_scores = np.mean(fold_per_class_f1, axis=0)
        
        # Log metrics to MLflow
        metrics_to_log = {
            'f1_candidate': mean_scores[0],
            'f1_confirmed': mean_scores[1],
            'f1_false_positive': mean_scores[2],
            'mean_f1': np.mean(mean_scores)
        }
        mlflow.log_metrics(metrics_to_log)
        
        trial.set_user_attr("mean_f1", metrics_to_log['mean_f1'])

    # Return the three F1 scores for Optuna's multi-objective optimization
    return mean_scores[0], mean_scores[1], mean_scores[2]


def main():
    print("--- Starting KOI-Specialist SOTA Search (v1.1 - Tunable Class Weights) ---")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    X = train_df.drop(columns=[TARGET_COLUMN])
    y = LabelEncoder().fit_transform(train_df[TARGET_COLUMN])
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # 2. Run Optuna Study within a parent MLflow run
    with mlflow.start_run(run_name="HPO_with_Tunable_Weights") as parent_run:
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("n_splits_cv", N_SPLITS)
        
        study = optuna.create_study(
            directions=['maximize', 'maximize', 'maximize'],
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(
            lambda trial: objective(trial, X, y, kf),
            n_trials=N_TRIALS,
            show_progress_bar=True
        )

        # 3. Process and Save Results
        pareto_front = study.best_trials
        results = []
        for t in pareto_front:
             results.append({
                "Trial": t.number,
                "f1_candidate": t.values[0],
                "f1_confirmed": t.values[1],
                "f1_false_positive": t.values[2],
                "mean_f1": t.user_attrs.get("mean_f1", 0),
                "Params": json.dumps(t.params)
            })
        
        results_df = pd.DataFrame(results).sort_values(by="mean_f1", ascending=False)
        
        print("\n--- Final Pareto Front (sorted by Mean F1) ---")
        print(results_df[['Trial', 'mean_f1', 'f1_candidate', 'f1_confirmed', 'f1_false_positive']].head())
        
        leaderboard_path = os.path.join(ARTIFACT_DIR, LEADERBOARD_FILE)
        results_df.to_csv(leaderboard_path, index=False)
        
        # Log the final leaderboard as an artifact in MLflow
        mlflow.log_artifact(leaderboard_path)
        
        # Log the best overall trial's params to the parent run for easy access
        best_trial = results_df.iloc[0].to_dict()
        best_params = json.loads(best_trial['Params'])
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_mean_f1", best_trial['mean_f1'])

    print(f"\n--- Search Complete ---")
    print(f"To view results, run 'mlflow ui' in your terminal and open http://12ystack-and-report.py'.0.1:5000")

if __name__ == '__main__':
    main()