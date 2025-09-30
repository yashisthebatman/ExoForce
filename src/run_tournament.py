# src/run_tournament.py
# PURPOSE: To run a multi-stage tournament to find the best specialists, mitigating subsampling risks.

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import warnings, os, json, ast, joblib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
STUDY_NAME = "ExoForge_SOTA_Tournament"
STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/exoforge_tournament_study.db"
LEADERBOARD_FILE = 'tournament_pareto_front.csv'
N_SPLITS = 3

# --- Tournament Configuration ---
# Rung 1: Blitz on a small sample to eliminate bad ideas
RUNG1_TRIALS = 200
RUNG1_SUBSAMPLE = 0.20
# Rung 2: Qualifiers for the survivors
RUNG2_SUBSAMPLE = 0.40
# Rung 3: Finals for the elite candidates
RUNG3_SUBSAMPLE = 0.80

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
    else:
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

def get_pareto_front_trials(study):
    """Helper to get the full trial objects from the Pareto front."""
    return study.best_trials

def run_tournament():
    print("--- Starting SOTA Tournament Search ---")
    
    # Load full training data
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    X_full = train_df.drop(columns=[TARGET_COLUMN])
    y_full = LabelEncoder().fit_transform(train_df[TARGET_COLUMN])

    # --- Rung 1: The Blitz ---
    print(f"\n--- Rung 1: Running {RUNG1_TRIALS} trials on {RUNG1_SUBSAMPLE*100:.0f}% of data ---")
    X_rung1, _, y_rung1, _ = train_test_split(X_full, y_full, train_size=RUNG1_SUBSAMPLE, stratify=y_full, random_state=1)
    study_rung1 = optuna.create_study(directions=['maximize', 'maximize', 'maximize'])
    study_rung1.optimize(lambda trial: objective(trial, X_rung1, y_rung1), n_trials=RUNG1_TRIALS, show_progress_bar=True)
    
    # Promote top 50% of candidates
    rung1_trials = get_pareto_front_trials(study_rung1)
    rung1_trials.sort(key=lambda t: np.mean(t.values), reverse=True)
    promoted_trials_rung2 = rung1_trials[:len(rung1_trials) // 2]
    print(f"--- Rung 1 Complete. Promoting {len(promoted_trials_rung2)} trials to Rung 2. ---")

    # --- Rung 2: The Qualifiers ---
    print(f"\n--- Rung 2: Evaluating {len(promoted_trials_rung2)} trials on {RUNG2_SUBSAMPLE*100:.0f}% of data ---")
    X_rung2, _, y_rung2, _ = train_test_split(X_full, y_full, train_size=RUNG2_SUBSAMPLE, stratify=y_full, random_state=2)
    study_rung2 = optuna.create_study(directions=['maximize', 'maximize', 'maximize'])
    for trial_to_promote in promoted_trials_rung2:
        study_rung2.enqueue_trial(trial_to_promote.params)
    study_rung2.optimize(lambda trial: objective(trial, X_rung2, y_rung2), n_trials=len(promoted_trials_rung2), show_progress_bar=True)
    
    # Promote top 50% of these
    rung2_trials = get_pareto_front_trials(study_rung2)
    rung2_trials.sort(key=lambda t: np.mean(t.values), reverse=True)
    promoted_trials_rung3 = rung2_trials[:len(rung2_trials) // 2]
    print(f"--- Rung 2 Complete. Promoting {len(promoted_trials_rung3)} trials to the Finals. ---")

    # --- Rung 3: The Finals ---
    print(f"\n--- Rung 3 (Finals): Evaluating {len(promoted_trials_rung3)} elite trials on {RUNG3_SUBSAMPLE*100:.0f}% of data ---")
    X_rung3, _, y_rung3, _ = train_test_split(X_full, y_full, train_size=RUNG3_SUBSAMPLE, stratify=y_full, random_state=3)
    study_final = optuna.create_study(
        directions=['maximize', 'maximize', 'maximize'],
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True # Allows resuming if it crashes
    )
    # Clear old trials if we are re-running
    for old_trial in study_final.trials:
        if old_trial.state != optuna.trial.TrialState.COMPLETE:
            study_final.sampler.remove_trial(old_trial._trial_id) # Internal API, use with care

    for trial_to_promote in promoted_trials_rung3:
        study_final.enqueue_trial(trial_to_promote.params)
    study_final.optimize(lambda trial: objective(trial, X_rung3, y_rung3), n_trials=len(promoted_trials_rung3), show_progress_bar=True)
    
    # --- Save Final Results ---
    print(f"\n--- Tournament Finished. ---")
    final_pareto = get_pareto_front_trials(study_final)
    
    results = []
    for trial in final_pareto:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                "Trial": trial.number,
                "f1_candidate": trial.values[0],
                "f1_confirmed": trial.values[1],
                "f1_false_positive": trial.values[2],
                "Params": str(trial.params)
            })
    
    results_df = pd.DataFrame(results)
    print("\n--- Final Pareto Front of Tournament Winners ---")
    print(results_df.head())
    
    results_df.to_csv(os.path.join(ARTIFACT_DIR, LEADERBOARD_FILE), index=False)
    print(f"\nTournament winners saved to '{LEADERBOARD_FILE}'.")
    print("\n--- Next Step: Run 'train_base_models.py' to build the diverse committee. ---")

if __name__ == '__main__':
    run_tournament()