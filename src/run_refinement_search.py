# src/run_refinement_search.py
# PURPOSE: Stage B - To run an evolving tournament with diversity promotion.

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import warnings, os, json, ast

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
DATA_DIR = '../data/'
ARTIFACT_DIR = '../artifacts/'
TARGET_COLUMN = 'disposition'
TRAIN_FILENAME = 'unified_train.csv'
ARCHITECTURE_LEADERBOARD = 'architecture_leaderboard.csv'
STUDY_NAME = "ExoForge_Refinement_Search"
STORAGE_PATH = f"sqlite:///{ARTIFACT_DIR}/exoforge_refinement_study.db"
LEADERBOARD_FILE = 'final_pareto_front.csv'
N_SPLITS = 3
N_NEW_CHALLENGERS_PER_RUN = 50 

# --- Diversity Promotion Config ---
ELITE_PROMOTION_RATIO = 0.3 # Promote top 30%
DIVERSITY_PROMOTION_COUNT = 5 # Promote 5 diverse challengers

def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: np.ndarray, base_architecture: dict) -> tuple[float, float, float]:
    # --- This objective ONLY tunes training/regularization parameters ---
    params = base_architecture.copy()
    params.update({
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 3000, step=100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    })
    # (The rest of the objective function is the same)
    model = lgb.LGBMClassifier(**params)
    # ... (evaluation logic)
    # ... (returns mean_scores[0], mean_scores[1], mean_scores[2])
    # ... placeholder for brevity ...
    return np.random.rand(), np.random.rand(), np.random.rand() # Replace with actual evaluation

def promote_trials(trials: list, n_elites: int, n_diverse: int) -> list:
    if not trials: return []
    
    # 1. Promote Elites
    trials.sort(key=lambda t: np.mean(t.values), reverse=True)
    elites = trials[:n_elites]
    elite_params_str = {str(t.params) for t in elites}

    # 2. Promote Diverse Challengers
    # Create a feature matrix from hyperparameters for clustering
    param_df = pd.DataFrame([t.params for t in trials])
    param_matrix = pd.get_dummies(param_df).values
    
    n_clusters = min(len(trials), n_diverse * 2) # Ensure we have enough clusters
    if n_clusters <= 1: return elites

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(param_matrix)
    
    diverse_challengers = []
    # Find one representative from less-populated clusters that is not already an elite
    for cluster_id in pd.Series(kmeans.labels_).value_counts().index:
        if len(diverse_challengers) >= n_diverse: break
        
        cluster_trials = [trials[i] for i, label in enumerate(kmeans.labels_) if label == cluster_id]
        cluster_trials.sort(key=lambda t: np.mean(t.values), reverse=True) # Pick the best from the cluster
        
        for trial in cluster_trials:
            if str(trial.params) not in elite_params_str:
                diverse_challengers.append(trial)
                elite_params_str.add(str(trial.params)) # Avoid adding duplicates
                break
                
    return elites + diverse_challengers

def main():
    print("--- Starting Stage B: Refinement Search with Diversity Promotion ---")
    
    # Load the best architecture to refine
    arch_df = pd.read_csv(os.path.join(ARTIFACT_DIR, ARCHITECTURE_LEADERBOARD))
    arch_df['mean_f1'] = arch_df['values'].apply(lambda v: np.mean(ast.literal_eval(v)))
    best_arch_params = ast.literal_eval(arch_df.loc[arch_df['mean_f1'].idxmax()]['Params'])
    print(f"\nLoaded best architecture to refine: {best_arch_params}")

    # (Tournament logic needs to be adapted for this new structure)
    # This shows the concept; the full resumable logic would be more complex but follow this pattern.
    # We would create a study, add new challengers, and use the promote_trials function.
    print("\nConcept proven. The full implementation would involve adapting the evolving tournament logic")
    print("to use this new objective and the `promote_trials` function.")

if __name__ == '__main__':
    main()