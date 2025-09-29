import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
import os

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = '../data/'
# IMPORTANT: Replace with the actual name of your target column
TARGET_COLUMN = 'disposition' 
# The positive label for your problem, often 'CONFIRMED'
POSITIVE_LABEL = 'CONFIRMED' 

def run_benchmark():
    """
    Loads data and runs a 5-fold cross-validation to establish a
    performance benchmark for the LightGBM model.
    """
    print("--- Starting ExoForge Benchmark ---")

    # --- 1. Load and Prepare Data ---
    try:
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'unified_train.csv'))
        validation_df = pd.read_csv(os.path.join(DATA_DIR, 'unified_validation.csv'))
        
        # For CV, we combine train and validation sets
        combined_df = pd.concat([train_df, validation_df], ignore_index=True)
        print(f"Combined training and validation data loaded. Shape: {combined_df.shape}")

    except FileNotFoundError as e:
        print(f"Error: Data files not found. Make sure 'unified_train.csv' and 'unified_validation.csv' are in the '{DATA_DIR}' directory.")
        print(e)
        return

    # Separate features (X) and target (y)
    X = combined_df.drop(columns=[TARGET_COLUMN])
    y_raw = combined_df[TARGET_COLUMN]

    # Encode target labels (e.g., 'CONFIRMED' -> 0, 'CANDIDATE' -> 1, etc.)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Store class names for later reporting
    class_names = le.classes_
    print(f"Target classes found: {class_names}")
    print(f"Class encoding: {list(zip(class_names, range(len(class_names))))}")


    # --- 2. Define Model and Cross-Validation ---
    
    # A solid, manually-tuned set of hyperparameters for LightGBM
    # These are reasonable defaults that we aim to beat.
    params = {
        'objective': 'multiclass',
        'num_class': len(class_names),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(combined_df), len(class_names)))
    fold_f1_scores = []

    print(f"\n--- Starting {N_SPLITS}-Fold Cross-Validation ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{N_SPLITS} ---")
        
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        # Predict probabilities for the validation set
        val_preds_proba = model.predict_proba(X_val)
        oof_preds[val_idx] = val_preds_proba

        # Calculate F1 score for this fold
        val_preds_labels = np.argmax(val_preds_proba, axis=1)
        # Using 'weighted' average for multiclass F1
        f1 = f1_score(y_val, val_preds_labels, average='weighted')
        fold_f1_scores.append(f1)
        print(f"Fold {fold+1} Weighted F1-Score: {f1:.5f}")

    # --- 3. Evaluate and Report ---
    print("\n--- Benchmark Cross-Validation Complete ---")
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    
    print(f"\nAverage Weighted F1-Score: {mean_f1:.5f} (+/- {std_f1:.5f})")

    # Overall Classification Report
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    print("\n--- Overall Classification Report (from OOF predictions) ---")
    string_class_names = [str(c) for c in class_names]
    print(classification_report(y, oof_pred_labels, target_names=string_class_names))

    print("\n--- Overall Confusion Matrix ---")
    print(pd.DataFrame(confusion_matrix(y, oof_pred_labels),
                       index=[f'True: {c}' for c in class_names],
                       columns=[f'Pred: {c}' for c in class_names]))
    
    print("\nBenchmark established. The AutoML engine must decisively beat these scores.")

if __name__ == '__main__':
    run_benchmark()