# src/stack_and_report.py
# FINAL VERSION - Correctly handles the 'source' column in the test set.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TEST_FILENAME = 'unified_test.csv'
DESCRIPTIVE_CLASS_NAMES = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

def main():
    print("--- Phase 3: Stacking Ensemble Finalization & Meta-Model Tuning ---")

    # --- 1. Load Artifacts ---
    print("Loading base models and out-of-fold predictions...")
    try:
        committee_of_experts = joblib.load(os.path.join(ARTIFACT_DIR, 'committee_of_experts.pkl'))
        oof_df = pd.read_csv(os.path.join(ARTIFACT_DIR, 'oof_predictions.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}"); print("Please run 'train_base_models.py' first."); return
        
    # --- 2. Tune the Meta-Model ---
    print("\n--- Tuning the meta-model on out-of-fold predictions... ---")
    X_meta_train, y_meta_train = oof_df.drop(columns='target'), oof_df['target']
    
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('model', LogisticRegression())])
    meta_model_search_space = [
        {'model': [LogisticRegression(random_state=42, max_iter=1000)], 'model__C': [0.1, 1.0, 10.0]},
        {'model': [lgb.LGBMClassifier(random_state=42)], 'model__n_estimators': [50, 100], 'model__max_depth': [2, 3], 'model__learning_rate': [0.05, 0.1]}
    ]
    grid_search = GridSearchCV(pipe, param_grid=meta_model_search_space, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_meta_train, y_meta_train)
    best_meta_model = grid_search.best_estimator_
    print(f"\nFound best meta-model with score {grid_search.best_score_:.4f}:\n{best_meta_model}")

    # --- 3. Generate Final Predictions on Test Set ---
    print("\nGenerating stacked predictions on the test set using the tuned meta-model...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILENAME))
    
    # --- THE FIX IS HERE ---
    # We must drop the non-numeric 'source' column from the test set features.
    X_test = test_df.drop(columns=[TARGET_COLUMN, 'source'], errors='ignore')
    
    le = LabelEncoder().fit(y_meta_train) 
    y_test = le.transform(test_df[TARGET_COLUMN])

    meta_features = [model.predict_proba(X_test) for name, model in committee_of_experts.items()]
    X_meta_test = np.concatenate(meta_features, axis=1)
    final_preds = best_meta_model.predict(X_meta_test)

    # --- 4. Final Evaluation and Reporting ---
    print("\n--- Final Stacked Ensemble Classification Report ---")
    print(classification_report(y_test, final_preds, target_names=DESCRIPTIVE_CLASS_NAMES))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, final_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=DESCRIPTIVE_CLASS_NAMES, yticklabels=DESCRIPTIVE_CLASS_NAMES)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    cm_path = os.path.join(ARTIFACT_DIR, 'confusion_matrix_stacked.png')
    plt.savefig(cm_path, bbox_inches='tight'); plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")
    
    # --- 5. Save the Full Stack ---
    final_stack = {'base_models': committee_of_experts, 'meta_model': best_meta_model.named_steps['model']}
    joblib.dump(final_stack, os.path.join(ARTIFACT_DIR, 'full_stack_model.pkl'))
    print(f"\nFull stacking pipeline saved to: {os.path.join(ARTIFACT_DIR, 'full_stack_model.pkl')}")

if __name__ == '__main__':
    main()