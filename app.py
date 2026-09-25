import streamlit as st
import os

st.set_page_config(page_title="CNC Sentinel", page_icon="🛠️", layout="wide")

st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0;">
    <h1 style="margin-bottom:0;">🛠️ CNC Sentinel</h1>
    <p style="font-size:1.1rem; color:#94A3B8; margin-top:0.2rem;">
        Plateforme de diagnostic prédictif pour fraiseuses CNC 3 axes
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "CNC Sentinel prédit la **durée de vie restante (RUL)** des composants "
    "critiques d'une fraiseuse CNC à partir de ses signaux capteurs, avant "
    "que la panne ne survienne — sur la base de deux modules construits "
    "avec la **même méthodologie standardisée**."
)

st.markdown("### Modules disponibles")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 🔧 Module 1 — Usure d'outil")
        st.markdown(
            "Prédiction du RUL de l'outil de coupe à partir des signaux de "
            "force, vibration et émission acoustique.\n\n"
            "*Dataset : PHM2010 Milling Data Challenge*"
        )
        st.metric("Précision de validation (R²)", "0.93 – 0.95")
        st.page_link("pages/1_Module_1_Usure_Outil.py", label="Ouvrir le Module 1 →", icon="🔧")

with col2:
    with st.container(border=True):
        st.markdown("#### ⚙️ Module 2 — Roulements")
        st.markdown(
            "Prédiction du RUL des roulements à partir des signaux vibratoires "
            "horizontal et vertical.\n\n"
            "*Dataset : FEMTO-ST / PRONOSTIA*"
        )
        module2_ready = os.path.exists("data/module2_model.pkl")
        st.metric("Statut du module", "Actif ✅" if module2_ready else "Bientôt disponible")
        st.page_link("pages/2_Module_2_Roulements.py", label="Ouvrir le Module 2 →", icon="⚙️")

st.divider()

with st.container(border=True):
    st.markdown("#### 🩺 Vue globale")
    st.markdown("Fusionne les deux modules en un score unique de santé machine.")
    st.page_link("pages/3_Sante_Globale.py", label="Voir la santé globale →", icon="🩺")

st.divider()

st.markdown(
    """
**Comment utiliser cette plateforme :**
1. Choisis un module ci-dessus, ou dans le menu à gauche
2. Teste avec des données de démonstration publiques (aucun import nécessaire), ou importe tes propres relevés capteurs
3. Obtiens un diagnostic instantané : RUL estimé, Health Index (0-100), et les facteurs qui expliquent la prédiction

*Système construit sur des jeux de données publics de référence — méthodologie reproductible et transférable à toute machine-outil 3 axes.*
"""
)
