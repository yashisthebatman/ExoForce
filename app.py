from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
import json
import threading
import mlflow
from mlflow.tracking import MlflowClient

# Imports needed for model loading and retraining
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Re-use your feature engineering functions
from src.create_dataset import calculate_physics_features, engineer_base_features

# --- App Configuration ---
app = Flask(__name__)

# --- Load Default Model and Report on Startup ---
try:
    STACKED_MODEL = joblib.load('artifacts/full_stack_model_koi.pkl')
    print("✅ Default stacked model loaded successfully.")
except FileNotFoundError:
    print("🔴 Default stacked model not found. Predictions with it will fail.")
    STACKED_MODEL = None

# Load the training data once for column reference and median imputation
try:
    TRAIN_DF = pd.read_csv('data/koi_train.csv')
    TRAIN_COLS = TRAIN_DF.drop(columns=['disposition']).columns
except FileNotFoundError:
    print("🔴 koi_train.csv not found. This is critical for feature engineering.")
    TRAIN_COLS = None
    TRAIN_DF = None

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # Explicitly set for clarity
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT_NAME = "ExoForge-UI-Retraining-Jobs"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
MLFLOW_REGISTERED_MODEL_NAME = "ExoForge-UI-Trained-Model"

# --- Core Logic Functions ---

def preprocess_input(data: dict) -> pd.DataFrame:
    """Takes raw user input and prepares it for any model."""
    df = pd.DataFrame([data])
    if TRAIN_DF is None:
        raise RuntimeError("Training data reference not loaded, cannot preprocess.")

    # Fill any missing essential columns with median values
    for col in TRAIN_DF.columns:
        if col not in df.columns and col != 'disposition':
            df[col] = TRAIN_DF[col].median()
    
    df_physics = calculate_physics_features(df)
    df_engineered = engineer_base_features(df_physics)
    
    final_df = pd.DataFrame(columns=TRAIN_COLS) 
    final_df = pd.concat([final_df, df_engineered], ignore_index=False)
    final_df = final_df[TRAIN_COLS]
    final_df.fillna(0, inplace=True)
    return final_df

def predict_with_stacked_model(features_df: pd.DataFrame):
    """Uses the default, pre-trained stacked model."""
    if STACKED_MODEL is None:
        raise RuntimeError("Default stacked model is not available.")
    model_dict = STACKED_MODEL
    meta_features = [
        base_model.predict_proba(features_df)
        for name, base_model in model_dict['base_models'].items()
    ]
    X_meta = np.concatenate(meta_features, axis=1)
    prediction_idx = model_dict['meta_model'].predict(X_meta)[0]
    prediction_proba = model_dict['meta_model'].predict_proba(X_meta)[0]
    
    # Standard class order for the stacked model
    return {
        'CANDIDATE': prediction_proba[0],
        'CONFIRMED': prediction_proba[1],
        'FALSE POSITIVE': prediction_proba[2]
    }, {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}[prediction_idx]


def predict_with_mlflow_model(model_uri: str, features_df: pd.DataFrame):
    """Loads a specific model from MLflow and uses it for prediction."""
    model = mlflow.lightgbm.load_model(model_uri)
    prediction_idx = model.predict(features_df)[0]
    prediction_proba = model.predict_proba(features_df)[0]
    
    # The order of probabilities depends on the model's `classes_` attribute
    # We must map it correctly to our standard names
    class_map = {0: 'CANDIDATE', 1: 'CONFIRMED', 2: 'FALSE POSITIVE'}
    probabilities = {}
    for i, class_label_idx in enumerate(model.classes_):
        class_name = class_map[class_label_idx]
        probabilities[class_name] = prediction_proba[i]

    return probabilities, class_map[prediction_idx]

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/galaxy/<path:filename>')
def galaxy_files(filename):
    return send_from_directory('galaxy', filename)

@app.route('/get_models', methods=['GET'])
def get_models():
    """Endpoint to fetch all available registered models from MLflow."""
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(MLFLOW_REGISTERED_MODEL_NAME, stages=["None", "Staging", "Production"])
        model_list = [{
            "version": v.version,
            "uri": v.source,
            "stage": v.current_stage
        } for v in versions]
        return jsonify(sorted(model_list, key=lambda x: int(x['version']), reverse=True))
    except Exception as e:
        print(f"🔴 Could not fetch models from MLflow: {e}")
        return jsonify({"error": "Could not connect to MLflow or model not found."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        model_uri = request_data.get('model_uri')
        input_data = request_data.get('features')
        
        print(f"Received prediction request for model: {model_uri or 'Default Stacked'}")
        
        processed_features = preprocess_input({k: float(v) for k, v in input_data.items()})
        print(f"Preprocessed features created with {processed_features.shape[1]} columns.")
        
        if model_uri == 'default' or not model_uri:
            probabilities, prediction = predict_with_stacked_model(processed_features)
        else:
            probabilities, prediction = predict_with_mlflow_model(model_uri, processed_features)

        return jsonify({'prediction': prediction, 'probabilities': probabilities})
    except Exception as e:
        print(f"🔴 Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"An internal error occurred: {e}"}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    params = request.get_json()
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['learning_rate'] = float(params['learning_rate'])
    params['num_leaves'] = int(params['num_leaves'])
    
    training_thread = threading.Thread(target=train_model_job, args=(params,))
    training_thread.start()
    return jsonify({'status': 'success', 'message': 'Model training started! Check MLflow UI for progress.'})

def train_model_job(params: dict):
    """Background job for model training and registration."""
    with mlflow.start_run(run_name="UI_Run"):
        mlflow.log_params(params)
        try:
            X_train = TRAIN_DF.drop(columns=['disposition'])
            y_train = TRAIN_DF['disposition']
            
            model = lgb.LGBMClassifier(random_state=42, **params)
            model.fit(X_train, y_train)
            
            # For simplicity, we'll log the training score. In a real scenario, you'd use a validation set.
            train_score = model.score(X_train, y_train)
            mlflow.log_metric("train_accuracy", train_score)
            
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name=MLFLOW_REGISTERED_MODEL_NAME
            )
            print(f"✅ Training job finished. Model registered as '{MLFLOW_REGISTERED_MODEL_NAME}'.")
        except Exception as e:
            print(f"🔴 Training job failed: {e}")

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # Run on a different port than MLflow
    app.run(debug=True, port=5001)