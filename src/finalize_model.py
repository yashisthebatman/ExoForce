import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
BENCHMARK_F1_SCORE = 0.80241
# Using the feature-rich datasets
TRAIN_FILENAME = 'unified_train.csv'
VALIDATION_FILENAME = 'unified_validation.csv'
TEST_FILENAME = 'unified_test.csv'

# --- PIPELINE UPGRADE 3: ENSEMBLE AVERAGING ---
N_ENSEMBLE_MODELS = 5 # Use the top 5 best models

def finalize_and_evaluate_ensemble():
    print("--- Starting Phase 4 (v2.0): Ensemble Finalization & Judgment ---")

    # --- 1. Load Data ---
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    validation_df = pd.read_csv(os.path.join(DATA_DIR, VALIDATION_FILENAME))
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILENAME))
    full_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    print("All datasets loaded successfully.")

    le = LabelEncoder()
    full_train_df[TARGET_COLUMN] = le.fit_transform(full_train_df[TARGET_COLUMN])
    test_df[TARGET_COLUMN] = le.transform(test_df[TARGET_COLUMN])
    class_names = [str(c) for c in le.classes_]

    X_train_full = full_train_df.drop(columns=[TARGET_COLUMN])
    y_train_full = full_train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    # --- 2. Load the Completed Optuna Study ---
    try:
        study_path = os.path.join(ARTIFACT_DIR, 'exoforge_study_v2.pkl')
        study = joblib.load(study_path)
        print(f"Optuna study loaded from {study_path}")
    except FileNotFoundError:
        print(f"Error: Could not find study object at {study_path}. Please run the optimizer first.")
        return

    # --- 3. Train and Predict with Top N Models ---
    # Get the best N trials (Optuna sorts them by best value)
    best_trials = study.best_trials[:N_ENSEMBLE_MODELS]
    print(f"\nTraining an ensemble of the top {len(best_trials)} models...")
    
    all_test_preds_proba = []

    for i, trial in enumerate(best_trials):
        print(f"  - Training model {i+1}/{len(best_trials)} (Trial {trial.number})...")
        params = trial.params
        params.update({
            'objective': 'multiclass',
            'num_class': len(class_names),
            'metric': 'multi_logloss',
            'seed': 42 + i, # Vary seed for diversity
            'n_jobs': -1,
            'verbose': -1,
            'class_weight': 'balanced'
        })
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_full, y_train_full)
        
        # Predict probabilities on the test set
        test_preds_proba = model.predict_proba(X_test)
        all_test_preds_proba.append(test_preds_proba)

    # --- 4. Average the Predictions ---
    print("\nAveraging predictions from all models...")
    # Shape: (n_models, n_samples, n_classes) -> (n_samples, n_classes)
    avg_preds_proba = np.mean(all_test_preds_proba, axis=0)
    
    # Get final labels by choosing the class with the highest average probability
    ensemble_preds = np.argmax(avg_preds_proba, axis=1)

    # --- 5. The Final Judgment ---
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
    print(classification_report(y_test, ensemble_preds, target_names=class_names))

if __name__ == '__main__':
    finalize_and_evaluate_ensemble()