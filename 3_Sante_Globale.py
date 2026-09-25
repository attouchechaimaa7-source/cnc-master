import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pipeline_utils import health_zone

st.set_page_config(page_title="Santé globale", page_icon="🩺", layout="wide")
st.title("🩺 Santé globale de la machine")
st.caption("Fusion des diagnostics Outil + Roulements en un score unique")

m1_ready = os.path.exists("data/module1_model.pkl")
m2_ready = os.path.exists("data/module2_model.pkl")

if not m1_ready:
    st.error("Module 1 indisponible.")
    st.stop()

m1_model = joblib.load("data/module1_model.pkl")
m1_features = joblib.load("data/module1_features.pkl")
m1_demo = pd.read_csv("data/module1_demo_data.csv")

st.markdown("Choisis un scénario de démonstration pour chaque composant :")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🔧 Outil de coupe")
    cutter = st.selectbox("Outil", sorted(m1_demo["experiment_tag"].unique()))
    sub1 = m1_demo[m1_demo.experiment_tag == cutter].sort_values("cut_id").reset_index(drop=True)
    cut_choice = st.slider("Avancement (n° de passe)", 1, int(sub1["cut_id"].max()), int(sub1["cut_id"].max()) // 2)
    row1 = sub1[sub1.cut_id == cut_choice].iloc[0]
    pred1 = float(m1_model.predict(row1[m1_features].values.reshape(1, -1))[0])
    health1 = float(np.clip(100 * pred1 / sub1["RUL"].max(), 0, 100))
    st.metric("Health Index — Outil", f"{health1:.0f} / 100", delta=health_zone(health1))

with col2:
    st.markdown("#### ⚙️ Roulement")
    if not m2_ready:
        st.info("Module 2 pas encore actif — le score global utilisera uniquement l'outil pour l'instant.")
        health2 = None
    else:
        m2_model = joblib.load("data/module2_model.pkl")
        m2_features = joblib.load("data/module2_features.pkl")
        if os.path.exists("data/module2_demo_data.csv"):
            m2_demo = pd.read_csv("data/module2_demo_data.csv")
            bearing = st.selectbox("Roulement", sorted(m2_demo["bearing"].unique()))
            sub2 = m2_demo[m2_demo.bearing == bearing].sort_values("file_index").reset_index(drop=True)
            idx_choice = st.slider("Avancement (roulement)", 0, int(sub2["file_index"].max()), int(sub2["file_index"].max()) // 2)
            row2 = sub2[sub2.file_index == idx_choice].iloc[0]
            health2 = float(m2_model.predict(row2[m2_features].values.reshape(1, -1))[0])
            st.metric("Health Index — Roulement", f"{health2:.0f} / 100", delta=health_zone(health2))
        else:
            health2 = None
            st.info("Pas de jeu de démo pour le Module 2.")

st.divider()

if health2 is not None:
    global_health = 0.5 * health1 + 0.5 * health2
else:
    global_health = health1

st.subheader("Score de santé global de la machine")
st.metric("Health Index global", f"{global_health:.0f} / 100", delta=health_zone(global_health))
st.progress(int(global_health))

st.caption(
    "Le score global pondère à parts égales les deux sous-systèmes surveillés. "
    "La pondération pourra être ajustée selon la criticité relative de chaque composant "
    "sur une machine réelle."
)
