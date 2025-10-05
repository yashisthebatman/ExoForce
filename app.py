from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
import json
import threading
import mlflow
from mlflow.tracking import MlflowClient
import lightgbm as lgb
from sklearn.metrics import classification_report

app = Flask(__name__)

# --- Load Models and Data References on Startup ---
try:
    STACKED_MODEL = joblib.load('artifacts/full_stack_model_koi.pkl')
    print("✅ Default stacked model loaded successfully.")
except FileNotFoundError:
    print("🔴 Default stacked model not found. Predictions will fail.")
    STACKED_MODEL = None

try:
    ENGINEERED_TRAIN_DF = pd.read_csv('data/koi_train.csv')
    ENGINEERED_VALIDATION_DF = pd.read_csv('data/koi_validation.csv')
    
    # We will use the original column names (with spaces) as our reference.
    ORIGINAL_COLS = ENGINEERED_TRAIN_DF.drop(columns=['disposition']).columns
    FEATURE_MEDIANS = ENGINEERED_TRAIN_DF[ORIGINAL_COLS].median().to_dict()
    
    print(f"✅ Engineered data references loaded. Expecting {len(ORIGINAL_COLS)} features.")

except Exception as e:
    print(f"🔴🔴🔴 CRITICAL ERROR ON STARTUP: {e} 🔴🔴🔴")
    STACKED_MODEL = None
    ENGINEERED_TRAIN_DF = None
    ORIGINAL_COLS = None
    FEATURE_MEDIANS = None

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT_NAME = "ExoForge-UI-Retraining-Jobs"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
MLFLOW_REGISTERED_MODEL_NAME = "ExoForge-UI-Trained-Model"


# --- Core Logic Functions ---
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Takes user input and creates a clean DataFrame with original (space-separated) column names.
    """
    if ORIGINAL_COLS is None or FEATURE_MEDIANS is None:
        raise RuntimeError("Data references not loaded, cannot preprocess.")

    filled_data = FEATURE_MEDIANS.copy()
    filled_data.update(data)
    
    # Return a DataFrame with the original column names in the correct order.
    df = pd.DataFrame([filled_data])
    return df[ORIGINAL_COLS]

def predict_with_stacked_model(features_df: pd.DataFrame):
    """
    Uses the default stacked model, handling different feature name requirements
    for different model types within the stack.
    """
    if STACKED_MODEL is None:
        raise RuntimeError("Default stacked model is not available.")
    model_dict = STACKED_MODEL
    
    meta_features = []
    for name, model in model_dict['base_models'].items():
        # --- DEFINITIVE FIX FOR PREDICTION ---
        # If the model is a LightGBM model, it needs sanitized (underscore) feature names.
        if 'lgbm' in name:
            df_sanitized = features_df.copy()
            df_sanitized.columns = df_sanitized.columns.str.replace(' ', '_', regex=False)
            # Ensure order matches the specific LGBM model
            df_sanitized = df_sanitized[model.feature_name_]
            meta_features.append(model.predict_proba(df_sanitized))
        # Otherwise (for the scikit-learn pipeline), use the original DataFrame with spaces.
        else:
            meta_features.append(model.predict_proba(features_df))
            
    X_meta = np.concatenate(meta_features, axis=1)
    prediction_idx = model_dict['meta_model'].predict(X_meta)[0]
    prediction_proba = model_dict['meta_model'].predict_proba(X_meta)[0]
    
    class_map = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}
    return {
        'CANDIDATE': prediction_proba[0],
        'CONFIRMED': prediction_proba[1],
        'FALSE POSITIVE': prediction_proba[2]
    }, class_map[prediction_idx]

# This function does not need changes, as it only deals with single LGBM models.
def predict_with_mlflow_model(model_uri: str, features_df: pd.DataFrame):
    model = mlflow.lightgbm.load_model(model_uri)
    df_sanitized = features_df.copy()
    df_sanitized.columns = df_sanitized.columns.str.replace(' ', '_', regex=False)
    df_sanitized = df_sanitized[model.feature_name_]
    
    prediction_idx = model.predict(df_sanitized)[0]
    prediction_proba = model.predict_proba(df_sanitized)[0]
    class_map = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}
    probabilities = {class_map[class_label_idx]: prob for i, (class_label_idx, prob) in enumerate(zip(model.classes_, prediction_proba))}
    return probabilities, class_map[prediction_idx]

# --- Flask Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/galaxy/<path:filename>')
def galaxy_files(filename): return send_from_directory('galaxy', filename)

@app.route('/get_models', methods=['GET'])
def get_models():
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(MLFLOW_REGISTERED_MODEL_NAME, stages=["None", "Staging", "Production"])
        model_list = [{"version": v.version, "uri": v.source, "stage": v.current_stage} for v in versions]
        return jsonify(sorted(model_list, key=lambda x: int(x['version']), reverse=True))
    except Exception as e: return jsonify({"error": f"Could not connect to MLflow: {e}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        input_features = {k: float(v) for k, v in request_data['features'].items() if v is not None and str(v).strip() != ''}
        processed_features = preprocess_input(input_features)
        
        model_uri = request_data.get('model_uri')
        if model_uri == 'default' or not model_uri:
            probabilities, prediction = predict_with_stacked_model(processed_features)
        else:
            probabilities, prediction = predict_with_mlflow_model(processed_features)

        return jsonify({'prediction': prediction, 'probabilities': probabilities})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f"An internal error occurred: {e}"}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    params = request.get_json()
    params = {k: float(v) for k, v in params.items()}
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['num_leaves'] = int(params['num_leaves'])
    
    threading.Thread(target=train_model_job, args=(params,)).start()
    return jsonify({'status': 'success', 'message': 'Model training started! Check MLflow UI.'})

def train_model_job(params: dict):
    """Background job for model training, evaluation, and registration."""
    # --- DEFINITIVE FIX FOR RETRAINING ---
    # Set the tracking URI and experiment *inside the thread* to establish context.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if ENGINEERED_TRAIN_DF is None:
        print("🔴 Cannot retrain: Engineered data not loaded.")
        return
        
    with mlflow.start_run(run_name="UI_Retraining_Run") as run:
        mlflow.log_params(params)
        try:
            X_train = ENGINEERED_TRAIN_DF.drop(columns=['disposition'])
            y_train = ENGINEERED_TRAIN_DF['disposition']
            model = lgb.LGBMClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            
            X_val = ENGINEERED_VALIDATION_DF.drop(columns=['disposition'])
            y_val = ENGINEERED_VALIDATION_DF['disposition']
            report = classification_report(y_val, model.predict(X_val), output_dict=True)
            
            mlflow.log_metric("validation_accuracy", report['accuracy'])
            mlflow.log_metric("validation_f1_weighted", report['weighted avg']['f1-score'])

            report_path = "validation_report.json"
            with open(report_path, 'w') as f: json.dump(report, f, indent=4)
            mlflow.log_artifact(report_path)
            os.remove(report_path)
            
            mlflow.lightgbm.log_model(lgb_model=model, artifact_path="model", registered_model_name=MLFLOW_REGISTERED_MODEL_NAME)
            print("✅ Training job finished successfully.")
        except Exception as e:
            print(f"🔴 Training job failed: {e}")
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    app.run(debug=True, port=5001)