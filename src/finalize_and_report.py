# ExoForge/src/finalize_and_report.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import ast
import shap
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
VALIDATION_FILENAME = 'unified_validation.csv'
TEST_FILENAME = 'unified_test.csv'

# --- Automated Ensembling Configuration ---
N_ENSEMBLE_MODELS = 5 # Use the top 5 models from the Pareto front

def predict_ensemble_proba(X, ensemble, le):
    """Averages predictions from all models in the ensemble."""
    all_probas = [model.predict_proba(X) for model in ensemble]
    avg_probas = np.mean(all_probas, axis=0)
    return avg_probas

def main():
    print("--- Starting ExoForge Automated Finalizer ---")
    
    # --- 1. Load Data ---
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    validation_df = pd.read_csv(os.path.join(DATA_DIR, VALIDATION_FILENAME))
    test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILENAME))
    full_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    
    le = LabelEncoder()
    full_train_df[TARGET_COLUMN] = le.fit_transform(full_train_df[TARGET_COLUMN])
    test_df[TARGET_COLUMN] = le.transform(test_df[TARGET_COLUMN])
    class_names = [str(c) for c in le.classes_]

    X_train_full = full_train_df.drop(columns=[TARGET_COLUMN])
    y_train_full = full_train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    print("Data loaded and prepared.")

    # --- 2. Build Champion Ensemble ---
    try:
        results_df = pd.read_csv(os.path.join(ARTIFACT_DIR, 'pareto_front_results.csv'))
        top_n_trials = results_df.head(N_ENSEMBLE_MODELS)
        print(f"\nBuilding ensemble from the top {N_ENSEMBLE_MODELS} models discovered...")
    except FileNotFoundError:
        print(f"Error: Could not find 'pareto_front_results.csv'. Please run the optimizer first.")
        return
        
    ensemble_models = []
    for index, trial_data in top_n_trials.iterrows():
        print(f"  - Training model for Trial {trial_data['Trial']}...")
        params = ast.literal_eval(trial_data['Params']) # Safely convert string of dict to dict
        params.update({'objective': 'multiclass', 'num_class': len(class_names), 'metric': 'multi_logloss', 'seed': 42+index, 'n_jobs':-1, 'verbose':-1})
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_full, y_train_full)
        ensemble_models.append(model)
        
    # --- 3. Save the Ensemble Artifact ---
    ensemble_artifact = {'models': ensemble_models, 'label_encoder': le}
    artifact_path = os.path.join(ARTIFACT_DIR, 'champion_ensemble_model.pkl')
    joblib.dump(ensemble_artifact, artifact_path)
    print(f"\nChampion ensemble model saved to: {artifact_path}")

    # --- 4. Deep Evaluation ---
    print("\n--- Performing Deep Evaluation on Test Set ---")
    ensemble_probas = predict_ensemble_proba(X_test, ensemble_models, le)
    ensemble_preds = np.argmax(ensemble_probas, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_preds, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, ensemble_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = os.path.join(ARTIFACT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, ensemble_probas, multi_class='ovr')
    print(f"\nROC AUC Score (One-vs-Rest): {roc_auc:.5f}")

    # --- 5. Interpretation with SHAP ---
    print("\n--- Performing SHAP Analysis for Interpretation ---")
    # SHAP works best on a subset of data for performance reasons
    X_shap = X_train_full.sample(100, random_state=42)
    
    # Explain the predictions of each model in the ensemble
    explainers = [shap.TreeExplainer(model) for model in ensemble_models]
    shap_values_list = [explainer.shap_values(X_shap) for explainer in explainers]

    # Average the SHAP values across the ensemble
    # For multiclass, shap_values_list is a list of (list of arrays)
    # We need to average them correctly
    avg_shap_values = np.mean(shap_values_list, axis=0)

    shap.summary_plot(avg_shap_values, X_shap, class_names=class_names, show=False)
    plt.title("SHAP Summary Plot (Ensemble)")
    shap_summary_path = os.path.join(ARTIFACT_DIR, 'shap_summary.png')
    plt.savefig(shap_summary_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to: {shap_summary_path}")

    print("\n--- Final Report Generation Complete ---")

if __name__ == '__main__':
    main()