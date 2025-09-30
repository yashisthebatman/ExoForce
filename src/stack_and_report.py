# src/stack_and_report.py
# PURPOSE: To train a meta-model on OOF predictions and generate final stacked results.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TEST_FILENAME = 'unified_test.csv'
DESCRIPTIVE_CLASS_NAMES = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']

def main():
    print("--- Starting Stacking Ensemble Finalization ---")

    # --- 1. Load Artifacts from Previous Step ---
    print("Loading base models and out-of-fold predictions...")
    try:
        lgbm = joblib.load(os.path.join(ARTIFACT_DIR, 'lgbm_base.pkl'))
        xgb = joblib.load(os.path.join(ARTIFACT_DIR, 'xgb_base.pkl'))
        cat = joblib.load(os.path.join(ARTIFACT_DIR, 'cat_base.pkl'))
        oof_df = pd.read_csv(os.path.join(ARTIFACT_DIR, 'oof_predictions.csv'))
    except FileNotFoundError as e:
        print(f"Error: Could not find base model artifacts. {e}")
        print("Please run 'train_base_models.py' first.")
        return
        
    base_models = [lgbm, xgb, cat]
    
    # --- 2. Train the Meta-Model (Level 1) ---
    print("Training the meta-model on out-of-fold predictions...")
    X_meta_train = oof_df.drop(columns='target')
    y_meta_train = oof_df['target']
    
    meta_model = LogisticRegression(random_state=42, C=1.0) # A simple, robust meta-model
    meta_model.fit(X_meta_train, y_meta_train)

    # --- 3. Generate Final Predictions on Test Set ---
    print("Generating stacked predictions on the test set...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILENAME))
    le = LabelEncoder().fit(test_df[TARGET_COLUMN]) # Fit encoder for label consistency
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = le.transform(test_df[TARGET_COLUMN])

    # Create the meta-features for the test set
    lgbm_test_preds = lgbm.predict_proba(X_test)
    xgb_test_preds = xgb.predict_proba(X_test)
    cat_test_preds = cat.predict_proba(X_test)
    X_meta_test = np.concatenate([lgbm_test_preds, xgb_test_preds, cat_test_preds], axis=1)

    # Get final stacked predictions and probabilities
    final_preds = meta_model.predict(X_meta_test)
    final_probas = meta_model.predict_proba(X_meta_test)

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

    roc_auc = roc_auc_score(y_test, final_probas, multi_class='ovr')
    print(f"\nROC AUC Score (One-vs-Rest): {roc_auc:.5f}")

    # --- 5. Save the Full Stack ---
    full_stack = {'base_models': base_models, 'meta_model': meta_model}
    joblib.dump(full_stack, os.path.join(ARTIFACT_DIR, 'full_stack_model.pkl'))
    print(f"\nFull stacking pipeline saved to: {os.path.join(ARTIFACT_DIR, 'full_stack_model.pkl')}")

    # --- 6. Interpretability ---
    print("\n--- Interpreting the Stack ---")
    meta_coef = meta_model.coef_
    feature_names = oof_df.columns.drop('target')
    
    print("Meta-Model Coefficients (How much it 'trusts' each base model's prediction for each class):")
    coef_df = pd.DataFrame(meta_coef, columns=feature_names, index=DESCRIPTIVE_CLASS_NAMES)
    print(coef_df)
    
    print("\nSHAP analysis on the strongest base model (LightGBM):")
    explainer = shap.TreeExplainer(lgbm)
    X_shap = X_test.sample(200, random_state=42)
    shap_values = explainer.shap_values(X_shap)
    shap.summary_plot(shap_values, X_shap, class_names=DESCRIPTIVE_CLASS_NAMES, show=False)
    plt.title("SHAP Summary Plot (LightGBM Base Model)")
    shap_path = os.path.join(ARTIFACT_DIR, 'shap_summary_stacked.png')
    plt.savefig(shap_path, bbox_inches='tight'); plt.close()
    print(f"SHAP plot saved to: {shap_path}")

if __name__ == '__main__':
    main()