"""
Fonctions d'extraction de features partagées entre les modules.
Même logique que dans les notebooks Module 1 et Module 2, pour garder
un pipeline cohérent entre l'entraînement (Colab) et le site (Streamlit).
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

MODULE1_SIGNAL_COLS = [
    'force_x', 'force_y', 'force_z',
    'vibration_x', 'vibration_y', 'vibration_z',
    'acoustic_emission_rms'
]

MODULE2_SIGNAL_COLS = ['horiz_accel', 'vert_accel']


def extract_features_module1(df_cut: pd.DataFrame) -> dict:
    """Features statistiques pour une passe de coupe (Module 1 - PHM2010)."""
    feat = {}
    for col in MODULE1_SIGNAL_COLS:
        x = df_cut[col].values
        feat[f'{col}_mean'] = np.mean(x)
        feat[f'{col}_std'] = np.std(x)
        feat[f'{col}_rms'] = np.sqrt(np.mean(x ** 2))
        feat[f'{col}_max'] = np.max(np.abs(x))
        feat[f'{col}_kurt'] = kurtosis(x)
        feat[f'{col}_skew'] = skew(x)
        feat[f'{col}_ptp'] = np.ptp(x)
    return feat


def extract_features_module2(df_snapshot: pd.DataFrame) -> dict:
    """Features statistiques pour un instantané vibratoire (Module 2 - FEMTO-ST)."""
    feat = {}
    for axis in MODULE2_SIGNAL_COLS:
        x = df_snapshot[axis].values.astype(float)
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        feat[f'{axis}_mean'] = np.mean(x)
        feat[f'{axis}_std'] = np.std(x)
        feat[f'{axis}_rms'] = rms
        feat[f'{axis}_peak'] = peak
        feat[f'{axis}_kurt'] = kurtosis(x)
        feat[f'{axis}_skew'] = skew(x)
        feat[f'{axis}_crest_factor'] = peak / rms if rms > 0 else 0
    return feat


def health_zone(score: float) -> str:
    if score >= 60:
        return "🟢 Sain"
    elif score >= 25:
        return "🟠 Surveillance"
    else:
        return "🔴 Alerte — maintenance requise"
