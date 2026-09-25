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


# Noms d'affichage professionnels (jamais les codes bruts du dataset à l'écran)
TOOL_DISPLAY_NAMES = {"c1": "Outil A", "c4": "Outil B", "c6": "Outil C"}


def display_name_for_tool(raw_tag: str) -> str:
    return TOOL_DISPLAY_NAMES.get(raw_tag, raw_tag)


def bearing_display_names(raw_values) -> dict:
    """Mappe dynamiquement chaque identifiant brut de roulement vers 'Roulement 1', 'Roulement 2', ..."""
    return {raw: f"Roulement {i + 1}" for i, raw in enumerate(sorted(set(raw_values)))}


def stage_label(rank: int, total: int) -> str:
    """Étiquette lisible pour une position dans une série de démo (indépendant du nombre de points)."""
    if total <= 1:
        return "Instantané unique"
    pct = rank / (total - 1)
    if pct <= 0.2:
        return "🟢 Début de vie"
    elif pct <= 0.75:
        return "🟠 Mi-vie"
    else:
        return "🔴 Fin de vie (usure avancée)"


def zone_color(score: float) -> str:
    if score >= 60:
        return "#22C55E"
    elif score >= 25:
        return "#F97316"
    else:
        return "#EF4444"


BASE_CSS = """
<style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .section-card {
        border: 1px solid #334155; border-radius: 14px; padding: 1.4rem 1.6rem;
        background: #1E293B; margin-bottom: 1rem;
    }
    .gauge-wrap {display:flex; flex-direction:column; align-items:center; margin: 0.5rem 0;}
    .gauge-label {margin-top:0.5rem; font-weight:600; color:#F1F5F9;}
    @keyframes pulseIcon {
        0% {transform: scale(1);} 50% {transform: scale(1.15);} 100% {transform: scale(1);}
    }
    .flow-icon {animation: pulseIcon 2.2s ease-in-out infinite;}
    .flow-track {position:relative; height:4px; background:#334155; border-radius:2px;
        margin: 0.3rem 6% 1.5rem 6%; overflow:hidden;}
    @keyframes sweep {from {left:-30%;} to {left:100%;}}
    .flow-progress {position:absolute; top:0; left:-30%; width:30%; height:100%;
        background:linear-gradient(90deg, transparent, #F97316, transparent);
        animation: sweep 2.4s linear infinite;}
</style>
"""


def inject_base_css():
    import streamlit as st
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def render_gauge_html(score: float, label: str, size: int = 130) -> str:
    color = zone_color(score)
    inner = int(size * 0.72)
    return f"""
    <div class="gauge-wrap">
        <div style="position:relative; width:{size}px; height:{size}px; border-radius:50%;
             background: conic-gradient({color} {score}%, #334155 {score}% 100%);
             display:flex; align-items:center; justify-content:center;">
            <div style="width:{inner}px; height:{inner}px; border-radius:50%; background:#0F172A;
                 display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:1.5rem; font-weight:700; color:{color};">{score:.0f}</div>
                <div style="font-size:0.65rem; color:#94A3B8;">/ 100</div>
            </div>
        </div>
        <div class="gauge-label">{label}</div>
    </div>
    """


def safe_feature_importance(model, feature_cols, top_n: int = 10):
    """Renvoie l'importance des features si le modèle l'expose, sinon None (au lieu de planter)."""
    if hasattr(model, "feature_importances_"):
        return (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
            .head(top_n)
        )
    return None
