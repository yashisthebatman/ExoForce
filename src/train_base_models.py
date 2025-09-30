# src/train_base_models.py
# PURPOSE: To build a DIVERSE committee using the tournament winners and a linear model.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import joblib, os, ast, warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
# --- Use the results from the new tournament optimizer ---
LEADERBOARD_FILE = 'tournament_pareto_front.csv'
N_SPLITS = 5

def main():
    print("--- Building the Diverse 'Tournament Winners' Committee for Stacking ---")

    # --- 1. Load Data ---
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    validation_df = pd.read_csv(os.path.join(DATA_DIR, 'unified_validation.csv'))
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)

    X = combined_df.drop(columns=[TARGET_COLUMN])
    y_raw = combined_df[TARGET_COLUMN]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # --- 2. Load the Tournament Leaderboard ---
    try:
        leaderboard = pd.read_csv(os.path.join(ARTIFACT_DIR, LEADERBOARD_FILE))
    except FileNotFoundError:
        print(f"Error: Could not find '{LEADERBOARD_FILE}'.")
        print("Please run 'run_tournament.py' first to find the best models.")
        return

    # --- 3. Select the Tuned LGBM Experts from the Leaderboard ---
    # Calculate a 'mean_f1' to find the best overall, balanced model
    leaderboard['mean_f1'] = leaderboard[['f1_candidate', 'f1_confirmed', 'f1_false_positive']].mean(axis=1)
    
    # Find the trial that had the best balanced performance
    best_balanced_trial = leaderboard.loc[leaderboard['mean_f1'].idxmax()]
    
    # Find the trial that was the best specialist for the most critical class
    best_confirmed_trial = leaderboard.loc[leaderboard['f1_confirmed'].idxmax()]
    
    # Define the committee of experts
    experts = {
        "lgbm_balanced": lgb.LGBMClassifier(**ast.literal_eval(best_balanced_trial['Params'])),
        "lgbm_confirmed_spec": lgb.LGBMClassifier(**ast.literal_eval(best_confirmed_trial['Params'])),
        # Add the fundamentally different linear model for maximum diversity
        "linear_model": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=42, max_iter=1000)
        )
    }
    
    print("\nSelected Committee of Experts:")
    print(f"  - lgbm_balanced (from Tournament Trial #{best_balanced_trial['Trial']})")
    print(f"  - lgbm_confirmed_spec (from Tournament Trial #{best_confirmed_trial['Trial']})")
    print(f"  - linear_model (for algorithmic diversity)")

    # --- 4. Generate Out-of-Fold (OOF) Predictions for Stacking ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    # The shape is (number of samples, 3 classes * number of expert models)
    oof_preds = np.zeros((len(X), 3 * len(experts)))
    final_models = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val = X.iloc[val_idx]
        
        for i, (name, model) in enumerate(experts.items()):
            # Train each expert on the training portion of the fold
            model.fit(X_train, y_train)
            start_col = i * 3
            # Predict ONLY on the validation portion of the fold to create unbiased OOF predictions
            oof_preds[val_idx, start_col:start_col+3] = model.predict_proba(X_val)

    # --- 5. Train Final Models on Full Data ---
    print("\nTraining final expert models on the full dataset for deployment...")
    for name, model in experts.items():
        final_models[name] = model.fit(X, y)

    # --- 6. Save Artifacts for the Final Stacking Stage ---
    # Save the trained models
    joblib.dump(final_models, os.path.join(ARTIFACT_DIR, 'committee_of_experts.pkl'))
    
    # Save the out-of-fold predictions which will become the training data for the meta-model
    oof_columns = [f'{name}_p{i}' for name in experts.keys() for i in range(3)]
    oof_df = pd.DataFrame(oof_preds, columns=oof_columns)
    oof_df['target'] = y
    oof_df.to_csv(os.path.join(ARTIFACT_DIR, 'oof_predictions.csv'), index=False)
    
    print("\nDiverse committee models and OOF predictions are ready.")
    print("--- Next Step: Run 'stack_and_report.py' to build the final SOTA model. ---")

if __name__ == '__main__':
    main()