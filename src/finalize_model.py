import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import optuna  # Import optuna
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
BENCHMARK_F1_SCORE = 0.80241
TRAIN_FILENAME = 'unified_train.csv'
VALIDATION_FILENAME = 'unified_validation.csv'
TEST_FILENAME = 'unified_test.csv'
DESCRIPTIVE_CLASS_NAMES = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

# --- NEW: Configuration to load the persistent study ---
STUDY_NAME = "ExoForge_Multi_Objective"
STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/exoforge_study.db"

N_ENSEMBLE_MODELS = 5

def finalize_and_evaluate_ensemble():
    print("--- Starting Phase 4 (v2.0): Ensemble Finalization & Judgment ---")

    # Data loading remains the same
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    validation_df = pd.read_csv(os.path.join(DATA_DIR, VALIDATION_FILENAME))
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILENAME))
    full_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    print("All datasets loaded successfully.")

    le = LabelEncoder()
    full_train_df[TARGET_COLUMN] = le.fit_transform(full_train_df[TARGET_COLUMN])
    test_df[TARGET_COLUMN] = le.transform(test_df[TARGET_COLUMN])
    
    class_names_for_report = DESCRIPTIVE_CLASS_NAMES if len(le.classes_) == len(DESCRIPTIVE_CLASS_NAMES) else [str(c) for c in le.classes_]

    X_train_full = full_train_df.drop(columns=[TARGET_COLUMN])
    y_train_full = full_train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # --- CHANGE: Load the completed Optuna study from the database ---
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
        print(f"Optuna study '{STUDY_NAME}' loaded from database with {len(study.trials)} trials.")
    except Exception as e:
        print(f"Error: Could not load study from {STORAGE_PATH}. Please run the optimizer first.")
        print(e)
        return

    # --- 3. Train and Predict with Top N Models ---
    best_trials = study.best_trials[:N_ENSEMBLE_MODELS]
    print(f"\nTraining an ensemble of the top {len(best_trials)} models...")
    
    all_test_preds_proba = []
    for i, trial in enumerate(best_trials):
        print(f"  - Training model {i+1}/{len(best_trials)} (Trial {trial.number})...")
        params = trial.params
        params.update({
            'objective': 'multiclass', 'num_class': len(le.classes_), 'metric': 'multi_logloss',
            'seed': 42 + i, 'n_jobs': -1, 'verbose': -1, 'class_weight': 'balanced'
        })
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_full, y_train_full)
        test_preds_proba = model.predict_proba(X_test)
        all_test_preds_proba.append(test_preds_proba)

    # --- 4. Average the Predictions & Judge ---
    print("\nAveraging predictions from all models...")
    avg_preds_proba = np.mean(all_test_preds_proba, axis=0)
    ensemble_preds = np.argmax(avg_preds_proba, axis=1)

    print("\n--- The Final Judgment (Ensemble Model) ---")
    final_f1_score = f1_score(y_test, ensemble_preds, average='weighted')
    print("\n--- Performance Comparison ---")
    print(f"Benchmark Model Weighted F1-Score:    {BENCHMARK_F1_SCORE:.5f}")
    print(f"Optimized Ensemble Model Weighted F1-Score: {final_f1_score:.5f}")

    if final_f1_score > BENCHMARK_F1_SCORE:
        improvement = ((final_f1_score - BENCHMARK_F1_SCORE) / BENCHMARK_F1_SCORE) * 100
        print(f"\nSUCCESS! The optimized ensemble beats the benchmark by {improvement:.2f}%.")
    else:
        print("\nNOTE: The optimized ensemble did not beat the benchmark score.")

    print("\n--- Final Ensemble Model Classification Report (on Test Set) ---")
    print(classification_report(y_test, ensemble_preds, target_names=class_names_for_report))

if __name__ == '__main__':
    finalize_and_evaluate_ensemble()