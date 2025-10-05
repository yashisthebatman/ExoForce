# ================================================
# FILE: src/create_dataset.py
# ================================================
# PURPOSE: KOI-FIRST STRATEGY IMPLEMENTATION.
# This script now focuses exclusively on the high-quality Kepler (KOI) dataset.
# It ingests only KOI.csv and applies a suite of advanced, physics-informed
# feature engineering specifically tailored for it.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
RAW_DATA_DIR = '../data_raw/'
OUTPUT_DATA_DIR = '../data/'
TARGET_MAP = {'CONFIRMED': 1, 'CANDIDATE': 0, 'FALSE POSITIVE': 2}

# --- Physical Constants ---
# Using constants improves readability and correctness for physics calculations.
PI = np.pi
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
R_SUN = 6.957e8  # Radius of the Sun in meters
M_SUN = 1.989e30 # Mass of the Sun in kg
SECONDS_PER_DAY = 86400

def calculate_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced, physics-informed features, including the critical
    stellar density discrepancy check.
    """
    print("Calculating advanced physics-informed features...")
    # Create a working copy to avoid SettingWithCopyWarning
    astro_df = df.copy()
    epsilon = 1e-6 # Small constant to avoid division by zero

    # --- 1. Calculate Stellar Density from Stellar Parameters (rho_star) ---
    # Convert stellar parameters from Solar units to SI units for calculation.
    srad_si = astro_df['koi_srad'] * R_SUN
    slogg_si = 10**astro_df['koi_slogg'] / 100 # Convert from log10(cgs) to m/s^2

    # Mass = g * R^2 / G
    smass_si = slogg_si * srad_si**2 / G
    # Density = Mass / Volume (Volume of a sphere = 4/3 * pi * R^3)
    rho_star = smass_si / ((4/3) * PI * srad_si**3)
    astro_df['rho_star'] = rho_star

    # --- 2. Calculate Stellar Density from Transit Shape (rho_circ) ---
    # This comes from Kepler's Third Law, assuming a circular orbit.
    period_si = astro_df['koi_period'] * SECONDS_PER_DAY
    duration_si = astro_df['koi_duration'] * 3600 # hours to seconds

    # A robust approximation for stellar density from transit params
    # This avoids using impact parameter which can be unreliable.
    rho_circ = (3 * PI / (G * period_si**2)) * ( ( (period_si / PI) * np.sqrt(astro_df['koi_depth']/1e6) ) / duration_si )**3
    astro_df['rho_circ'] = rho_circ

    # --- 3. The "Money" Feature: Density Discrepancy ---
    # A value far from 1.0 is a strong indicator of a false positive.
    astro_df['density_discrepancy_ratio'] = astro_df['rho_star'] / (astro_df['rho_circ'] + epsilon)

    return astro_df


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard feature engineering: log transforms, ratios, and interactions.
    """
    print("Applying base feature engineering (logs, ratios, interactions)...")
    rich_df = df.copy()
    epsilon = 1e-6

    # Log transforms for heavily skewed features (as confirmed by EDA)
    for col in ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_insol', 'koi_model_snr']:
        rich_df[f'log_{col}'] = np.log1p(rich_df[col])

    # Error ratios to represent relative uncertainty
    for err_col in ['koi_period_err1', 'koi_duration_err1', 'koi_prad_err1', 'koi_steff_err1']:
         # The raw data uses _err1 for positive error
        base_col = err_col.replace('_err1', '')
        if base_col in rich_df.columns:
            rich_df[f'{base_col}_err_ratio'] = rich_df[err_col] / (rich_df[base_col] + epsilon)

    # Simple physics-based ratios
    rich_df['v_shape_metric'] = rich_df['koi_depth'] / (rich_df['koi_duration'] + epsilon)
    rich_df['planet_star_radius_ratio'] = rich_df['koi_prad'] / (rich_df['koi_srad'] + epsilon)

    # Interaction with key false positive flags
    rich_df['depth_x_fpflag_co'] = rich_df['koi_depth'] * rich_df['koi_fpflag_co']
    rich_df['snr_x_fpflag_nt'] = rich_df['koi_model_snr'] * rich_df['koi_fpflag_nt']
    
    # Polynomial features on a small subset of the most powerful predictors
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features = poly.fit_transform(rich_df[['koi_score', 'koi_model_snr', 'koi_depth']])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['koi_score', 'koi_model_snr', 'koi_depth']))
    rich_df = pd.concat([rich_df.reset_index(drop=True), poly_df], axis=1)

    return rich_df


if __name__ == '__main__':
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    print("--- Creating KOI-Only 'Purist' Datasets ---")

    # 1. Load Data (KOI only)
    try:
        koi_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'KOI.csv'), comment='#')
        print(f"Successfully loaded KOI.csv with shape: {koi_df.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'KOI.csv' is in the '../data_raw/' directory.")
        exit()

    # 2. Basic Cleaning
    koi_df = koi_df.dropna(subset=['koi_disposition'])
    koi_df['disposition'] = koi_df['koi_disposition'].map(TARGET_MAP)
    # Median imputation for simplicity, as done previously
    for col in koi_df.select_dtypes(include=np.number).columns:
        if koi_df[col].isnull().any():
            koi_df[col] = koi_df[col].fillna(koi_df[col].median())

    # 3. Apply Feature Engineering Pipeline
    astro_df = calculate_physics_features(koi_df)
    final_df = engineer_base_features(astro_df)

    # 4. Final Cleanup
    # Drop original columns that have been transformed or are no longer needed
    # Also drop ra/dec as per our purist philosophy
    cols_to_drop = [
        'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition',
        'koi_tce_delivname', 'ra', 'dec',
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_insol', 'koi_model_snr', # Originals
        'koi_period_err1', 'koi_duration_err1', 'koi_prad_err1', 'koi_steff_err1', # Error columns
        'koi_period_err2', 'koi_duration_err2', 'koi_prad_err2', 'koi_steff_err2',
        'koi_time0bk_err1', 'koi_time0bk_err2', 'koi_impact_err1', 'koi_impact_err2',
        'koi_depth_err1', 'koi_depth_err2', 'koi_teq_err1', 'koi_teq_err2',
        'koi_insol_err1', 'koi_insol_err2', 'koi_steff_err2', 'koi_slogg_err1',
        'koi_slogg_err2', 'koi_srad_err1', 'koi_srad_err2'
    ]
    final_df = final_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Drop any remaining non-numeric columns except the target
    final_df = final_df.select_dtypes(include=np.number)

    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(inplace=True)
    print(f"Final feature-engineered dataset has shape: {final_df.shape}")

    # 5. Split and Save Data
    X = final_df.drop('disposition', axis=1)
    y = final_df['disposition']

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val)

    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(OUTPUT_DATA_DIR, 'koi_train.csv'), index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(OUTPUT_DATA_DIR, 'koi_validation.csv'), index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(OUTPUT_DATA_DIR, 'koi_test.csv'), index=False)

    print("\n--- New 'KOI-Only' datasets created successfully! ---")
    print(f"Train set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")