# src/train_base_models.py
# Updated to correctly handle the new 'source' column in the dataset.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import joblib, os, json, warnings

warnings.filterwarnings('ignore')

# --- Configuration (Unchanged) ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
VALIDATION_FILENAME = 'unified_validation.csv'
LEADERBOARD_FILE = 'sota_pareto_front_specialists.csv'
N_SPLITS = 5

# --- Specialist Selection Function (Unchanged) ---
def select_diverse_specialists(leaderboard: pd.DataFrame) -> dict:
    # ... (This function remains exactly the same as before)
    if leaderboard.empty: raise ValueError("Leaderboard empty.")
    selected_trials, used_indices = {}, set()
    leaderboard['mean_f1'] = leaderboard[['f1_candidate', 'f1_confirmed', 'f1_false_positive']].mean(axis=1)
    best_all_rounder_idx = leaderboard['mean_f1'].idxmax()
    selected_trials['lgbm_all_rounder'] = leaderboard.loc[best_all_rounder_idx]; used_indices.add(best_all_rounder_idx)
    leaderboard['spec_candidate'] = leaderboard['f1_candidate'] - leaderboard[['f1_confirmed', 'f1_false_positive']].mean(axis=1)
    leaderboard['spec_confirmed'] = leaderboard['f1_confirmed'] - leaderboard[['f1_candidate', 'f1_false_positive']].mean(axis=1)
    leaderboard['spec_fp'] = leaderboard['f1_false_positive'] - leaderboard[['f1_candidate', 'f1_confirmed']].mean(axis=1)
    for role, spec_col in [('lgbm_confirmed_spec', 'spec_confirmed'), ('lgbm_candidate_spec', 'spec_candidate'), ('lgbm_fp_spec', 'spec_fp')]:
        available = leaderboard.drop(index=list(used_indices));
        if available.empty: break
        best_spec_idx = available[spec_col].idxmax()
        selected_trials[role] = leaderboard.loc[best_spec_idx]; used_indices.add(best_spec_idx)
    return selected_trials

def main():
    print("--- Phase 2: Building the Diverse 'Specialist Committee' ---")

    # --- Load Data ---
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    validation_df = pd.read_csv(os.path.join(DATA_DIR, VALIDATION_FILENAME))
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)

    # --- MODIFICATION: Drop the 'source' column before training ---
    X = combined_df.drop(columns=[TARGET_COLUMN, 'source'], errors='ignore')
    y = LabelEncoder().fit_transform(combined_df[TARGET_COLUMN])
    
    # --- The rest of the script is unchanged ---
    leaderboard = pd.read_csv(os.path.join(ARTIFACT_DIR, LEADERBOARD_FILE))
    diverse_committee_trials = select_diverse_specialists(leaderboard.copy())
    experts = {}
    print("\nSelected Diverse Committee of Experts:")
    for name, trial_info in diverse_committee_trials.items():
        print(f"  - {name} (from Trial #{int(trial_info['Trial'])})")
        experts[name] = lgb.LGBMClassifier(**json.loads(trial_info['Params']))
    experts["linear_model"] = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000, C=1.0))
    print("  - linear_model (for algorithmic diversity)")

    # --- OOF Generation and Saving (Unchanged) ---
    print("\nGenerating out-of-fold (OOF) predictions...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    num_classes = len(np.unique(y))
    oof_preds = np.zeros((len(X), num_classes * len(experts)))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train, X_val = X.iloc[train_idx], y[train_idx], X.iloc[val_idx]
        for i, (name, model) in enumerate(experts.items()):
            model.fit(X_train, y_train)
            oof_preds[val_idx, i*num_classes:(i+1)*num_classes] = model.predict_proba(X_val)

    final_models = {name: model.fit(X, y) for name, model in experts.items()}
    joblib.dump(final_models, os.path.join(ARTIFACT_DIR, 'committee_of_experts.pkl'))
    oof_columns = [f'{name}_p{i}' for name in experts.keys() for i in range(num_classes)]
    oof_df = pd.DataFrame(oof_preds, columns=oof_columns); oof_df['target'] = y
    oof_df.to_csv(os.path.join(ARTIFACT_DIR, 'oof_predictions.csv'), index=False)
    
    print("\n--- Next Step: Run 'stack_and_report.py' ---")

if __name__ == '__main__':
    main()