# src/create_dataset.py
# ULTIMATE PURIST VERSION - Creates the full suite of engineered features, drops ra/dec,
# and ADDS a 'source' column for robust, source-based cross-validation.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
RAW_DATA_DIR = '../data_raw/' 
OUTPUT_DATA_DIR = '../data/' 
COLUMN_MAPPING = {
    'disposition': ['koi_disposition', 'tfopwg_disp', 'disposition'], 'period': ['koi_period', 'pl_orbper', 'pl_orbper'],
    'duration': ['koi_duration', 'pl_trandurh', 'pl_trandur'], 'depth': ['koi_depth', 'pl_trandep', 'pl_trandep'],
    'planet_radius': ['koi_prad', 'pl_rade', 'pl_rade'], 'equilibrium_temp': ['koi_teq', 'pl_eqt', 'pl_eqt'],
    'insolation_flux': ['koi_insol', 'pl_insol', 'pl_insol'], 'stellar_temp': ['koi_steff', 'st_teff', 'st_teff'],
    'stellar_gravity': ['koi_slogg', 'st_logg', 'st_logg'], 'stellar_radius': ['koi_srad', 'st_rad', 'st_rad'],
    'snr': ['koi_model_snr', None, None], 'score': ['koi_score', None, None],
    'planet_count': ['koi_tce_plnt_num', None, 'sy_pnum'], 'fpflag_nt': ['koi_fpflag_nt', None, None],
    'fpflag_ss': ['koi_fpflag_ss', None, None], 'fpflag_co': ['koi_fpflag_co', None, None],
    'fpflag_ec': ['koi_fpflag_ec', None, None], 'period_err': ['koi_period_err1', 'pl_orbpererr1', 'pl_orbpererr1'],
    'prad_err': ['koi_prad_err1', 'pl_radeerr1', 'pl_radeerr1'],
    'duration_err': ['koi_duration_err1', 'pl_trandurherr1', 'pl_trandurerr1'],
    'steff_err': ['koi_steff_err1', 'st_tefferr1', 'st_tefferr1'], 'ra': ['ra', 'ra', 'ra'], 'dec': ['dec', 'dec', 'dec'],
}
TARGET_MAP = {'CONFIRMED': 1, 'CANDIDATE': 0, 'FALSE POSITIVE': 2, 'CP': 1, 'PC': 0, 'FP': 2}

def unify_and_clean_raw_data():
    all_dfs = []
    for source_name, filename in [('koi', 'KOI.csv'), ('toi', 'TOI.csv'), ('k2', 'K2.csv')]:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, filename), comment='#')
        unified_df = pd.DataFrame()
        source_idx = ['koi', 'toi', 'k2'].index(source_name)
        for canonical, source_names in COLUMN_MAPPING.items():
            source_col = source_names[source_idx]
            if source_col and source_col in df.columns: unified_df[canonical] = df[source_col]
        
        # --- NEW: Add the source identifier ---
        unified_df['source'] = source_name
            
        if source_name == 'toi' and 'duration' in unified_df.columns: unified_df['duration'] /= 24.0
        if source_name == 'toi' and 'duration_err' in unified_df.columns: unified_df['duration_err'] /= 24.0
        if 'disposition' in unified_df.columns: unified_df['disposition'] = unified_df['disposition'].str.upper().map(TARGET_MAP)
        all_dfs.append(unified_df)
        
    master_df = pd.concat(all_dfs, ignore_index=True).dropna(subset=['disposition'])
    master_df['disposition'] = master_df['disposition'].astype(int)
    for col in master_df.select_dtypes(include=np.number).columns:
        if master_df[col].isnull().any(): master_df[col] = master_df[col].fillna(master_df[col].median())
    return master_df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Applying the 'Purist' (scientific) feature engineering plan...")
    rich_df, epsilon = df.copy(), 1e-6
    for col in ['period', 'depth', 'duration', 'planet_radius', 'insolation_flux']: rich_df[f'log_{col}'] = np.log1p(rich_df[col])
    for col in ['period_err', 'prad_err', 'duration_err', 'steff_err']:
        original_col = col.replace('_err', '')
        if col in rich_df.columns and original_col in rich_df.columns: rich_df[f'{original_col}_err_ratio'] = rich_df[col] / (rich_df[original_col] + epsilon)
    rich_df['normalized_depth'] = rich_df['depth'] / (rich_df['stellar_radius']**2 + epsilon)
    rich_df['V_shape_metric'] = rich_df['depth'] / (rich_df['duration'] + epsilon)
    rich_df['planet_star_radius_ratio'] = rich_df['planet_radius'] / (rich_df['stellar_radius'] + epsilon)
    rich_df['depth_x_fpflag_co'] = rich_df['depth'] * rich_df['fpflag_co']
    rich_df['snr_x_fpflag_nt'] = rich_df['snr'] * rich_df['snr']
    cols_to_drop = ['period', 'depth', 'duration', 'planet_radius', 'insolation_flux', 'period_err', 'prad_err', 'duration_err', 'steff_err', 'ra', 'dec']
    rich_df = rich_df.drop(columns=cols_to_drop, errors='ignore')
    return rich_df

if __name__ == '__main__':
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    print("--- Creating Final 'Ultimate Purist' Datasets with Source ID ---")
    try: master_df = unify_and_clean_raw_data()
    except FileNotFoundError as e: print(f"Error: {e}"); exit()
    rich_df = engineer_features(master_df)
    rich_df = rich_df.loc[:, (rich_df != rich_df.iloc[0]).any()]
    rich_df.replace([np.inf, -np.inf], np.nan, inplace=True); rich_df.dropna(inplace=True)
    X, y = rich_df.drop('disposition', axis=1), rich_df['disposition']
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)
    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(OUTPUT_DATA_DIR, 'unified_train.csv'), index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(OUTPUT_DATA_DIR, 'unified_validation.csv'), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(OUTPUT_DATA_DIR, 'unified_test.csv'), index=False)
    print("\n--- New 'Ultimate Purist' datasets created and saved successfully! ---")