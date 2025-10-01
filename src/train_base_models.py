# src/train_base_models.py
# PURPOSE: To build a DIVERSE committee of experts for stacking. It selects specialists
#          from the tournament results in a way that guarantees model diversity and then
#          generates the out-of-fold (OOF) predictions needed for the meta-model.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import os
import json
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
VALIDATION_FILENAME = 'unified_validation.csv' # Added for a more robust training set
# --- Use the results from the new SOTA specialist search ---
LEADERBOARD_FILE = 'sota_pareto_front_specialists.csv'
N_SPLITS = 5 # Use 5 folds for robust OOF generation

def select_diverse_specialists(leaderboard: pd.DataFrame) -> dict:
    """
    Selects a diverse committee of specialists from the leaderboard, ensuring that
    the chosen models are distinct from each other to maximize ensemble diversity.
    
    Returns:
        A dictionary where keys are the role (e.g., 'lgbm_all_rounder') and
        values are the corresponding row from the leaderboard DataFrame.
    """
    if leaderboard.empty:
        raise ValueError("Leaderboard is empty. Cannot select specialists.")

    selected_trials = {}
    used_trial_indices = set()

    # 1. Select the Best All-Rounder (highest mean F1)
    leaderboard['mean_f1'] = leaderboard[['f1_candidate', 'f1_confirmed', 'f1_false_positive']].mean(axis=1)
    best_all_rounder_idx = leaderboard['mean_f1'].idxmax()
    selected_trials['lgbm_all_rounder'] = leaderboard.loc[best_all_rounder_idx]
    used_trial_indices.add(best_all_rounder_idx)
    
    # 2. Define specialization scores to find models that excel at one class relative to others
    leaderboard['spec_candidate'] = leaderboard['f1_candidate'] - leaderboard[['f1_confirmed', 'f1_false_positive']].mean(axis=1)
    leaderboard['spec_confirmed'] = leaderboard['f1_confirmed'] - leaderboard[['f1_candidate', 'f1_false_positive']].mean(axis=1)
    leaderboard['spec_fp'] = leaderboard['f1_false_positive'] - leaderboard[['f1_candidate', 'f1_confirmed']].mean(axis=1)

    # 3. Iteratively select the best specialist for each role from the REMAINING trials
    specialist_roles = {
        'lgbm_confirmed_spec': 'spec_confirmed',
        'lgbm_candidate_spec': 'spec_candidate',
        'lgbm_fp_spec': 'spec_fp',
    }

    for role, spec_col in specialist_roles.items():
        # Create a temporary board of available candidates (those not yet selected)
        available_candidates = leaderboard.drop(index=list(used_trial_indices))
        if available_candidates.empty:
            print(f"Warning: Ran out of unique models to select for role '{role}'.")
            break
        
        # Find the best specialist among the available ones
        best_spec_idx = available_candidates[spec_col].idxmax()
        selected_trials[role] = leaderboard.loc[best_spec_idx]
        used_trial_indices.add(best_spec_idx)
        
    return selected_trials

def main():
    print("--- Phase 2: Building the Diverse 'Specialist Committee' for Stacking ---")

    # --- 1. Load Data ---
    # Combine train and validation for a larger, more robust dataset for OOF generation
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
        validation_df = pd.read_csv(os.path.join(DATA_DIR, VALIDATION_FILENAME))
        combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure data files are in '../data/'.")
        return

    X = combined_df.drop(columns=[TARGET_COLUMN])
    y_raw = combined_df[TARGET_COLUMN]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # --- 2. Load the Tournament Leaderboard ---
    try:
        leaderboard = pd.read_csv(os.path.join(ARTIFACT_DIR, LEADERBOARD_FILE))
    except FileNotFoundError:
        print(f"Error: Could not find '{LEADERBOARD_FILE}'.")
        print("Please run 'run_sota_search.py' first to find the best models.")
        return

    # --- 3. Select the Diverse Committee of Experts ---
    diverse_committee_trials = select_diverse_specialists(leaderboard.copy())
    
    experts = {}
    print("\nSelected Diverse Committee of Experts:")
    for name, trial_info in diverse_committee_trials.items():
        print(f"  - {name} (from Tournament Trial #{int(trial_info['Trial'])})")
        # Use json.loads because params are now saved as a proper JSON string
        params = json.loads(trial_info['Params'])
        experts[name] = lgb.LGBMClassifier(**params)
        
    # Add the fundamentally different linear model for maximum algorithmic diversity
    experts["linear_model"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    )
    print("  - linear_model (for algorithmic diversity)")

    # --- 4. Generate Out-of-Fold (OOF) Predictions for Stacking ---
    print("\nGenerating out-of-fold (OOF) predictions for the meta-model...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    num_classes = len(np.unique(y))
    oof_preds = np.zeros((len(X), num_classes * len(experts)))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Processing Fold {fold+1}/{N_SPLITS} ---")
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]
        
        for i, (name, model) in enumerate(experts.items()):
            # Train each expert on the training portion of the fold
            model.fit(X_train, y_train)
            start_col = i * num_classes
            # Predict ONLY on the validation portion of the fold to create unbiased OOF predictions
            oof_preds[val_idx, start_col:start_col+num_classes] = model.predict_proba(X_val)

    # --- 5. Train Final Models on Full Data for Deployment ---
    print("\nTraining final expert models on the full dataset...")
    final_models = {}
    for name, model in experts.items():
        final_models[name] = model.fit(X, y)

    # --- 6. Save Artifacts for the Final Stacking Stage ---
    # Save the final, fully trained committee models
    joblib.dump(final_models, os.path.join(ARTIFACT_DIR, 'committee_of_experts.pkl'))
    
    # Save the out-of-fold predictions, which will be the training data for the meta-model
    oof_columns = [f'{name}_p{i}' for name in experts.keys() for i in range(num_classes)]
    oof_df = pd.DataFrame(oof_preds, columns=oof_columns)
    oof_df['target'] = y
    oof_df.to_csv(os.path.join(ARTIFACT_DIR, 'oof_predictions.csv'), index=False)
    
    print("\nDiverse committee models and OOF predictions have been saved.")
    print("--- Next Step: Run 'stack_and_report.py' to build and evaluate the final SOTA model. ---")

if __name__ == '__main__':
    main()