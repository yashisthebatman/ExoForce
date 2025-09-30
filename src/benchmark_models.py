# src/benchmark_models.py
# PURPOSE: To establish a baseline for the two DIVERSE model types in our final stack.

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import warnings
import os

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
VALIDATION_FILENAME = 'unified_validation.csv'
DESCRIPTIVE_CLASS_NAMES = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
N_SPLITS = 5

def evaluate_model(model_name, model, X, y):
    """Evaluates a given model using cross-validation and reports per-class F1 scores."""
    print(f"\n--- Benchmarking {model_name} ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    per_class_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        per_class_f1_scores.append(f1_score(y_val, preds, average=None, labels=range(len(DESCRIPTIVE_CLASS_NAMES))))

    mean_scores = np.mean(per_class_f1_scores, axis=0)
    print("Average Class F1-Scores:")
    for i, class_name in enumerate(DESCRIPTIVE_CLASS_NAMES):
        print(f"  - {class_name}: {mean_scores[i]:.4f}")
    
    return mean_scores

def main():
    print("--- Starting Diverse Model Benchmark ---")
    
    # Load and combine data for a robust benchmark
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILENAME))
    validation_df = pd.read_csv(os.path.join(DATA_DIR, VALIDATION_FILENAME))
    combined_df = pd.concat([train_df, validation_df], ignore_index=True)
    
    X = combined_df.drop(columns=[TARGET_COLUMN])
    y_raw = combined_df[TARGET_COLUMN]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Define the two diverse models for our final ensemble
    models = {
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        
        # Linear models require scaled data for optimal performance.
        # We use a scikit-learn pipeline to bundle scaling and modeling.
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=42, max_iter=1000)
        )
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X, y)
        
    results_df = pd.DataFrame(results, index=DESCRIPTIVE_CLASS_NAMES).T
    print("\n--- Overall Benchmark Comparison ---")
    print(results_df)
    print("\nThis table shows the natural strengths of each model before tuning.")
    print("Notice their different performance profiles, which makes them ideal for stacking.")

if __name__ == '__main__':
    main()