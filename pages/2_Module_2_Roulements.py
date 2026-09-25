import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pipeline_utils import (
    MODULE2_SIGNAL_COLS, extract_features_module2, health_zone,
    bearing_display_names, stage_label, safe_feature_importance,
)

st.set_page_config(page_title="Module 2 — Roulements", page_icon="⚙️", layout="wide")
st.title("⚙️ Module 2 — RUL Roulements")
st.caption("Prédiction de la durée de vie restante d'un roulement à partir de sa signature vibratoire")

MODEL_PATH = "data/module2_model.pkl"
FEATURES_PATH = "data/module2_features.pkl"
DEMO_PATH = "data/module2_demo_data.csv"

if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)):
    st.warning(
        "Le modèle du Module 2 n'est pas encore chargé dans cette instance. "
        "Dépose `module2_model.pkl`, `module2_features.pkl` et (idéalement) "
        "`module2_demo_data.csv` dans le dossier `data/` du dépôt, puis recharge la page."
    )
    st.stop()


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, feature_cols


@st.cache_data
def load_demo_data():
    if os.path.exists(DEMO_PATH):
        return pd.read_csv(DEMO_PATH)
    return None


model, FEATURE_COLS = load_model()
demo = load_demo_data()

mode = st.radio("Mode", ["Démonstration", "Importer mes données"], horizontal=True)

if mode == "Démonstration":
    if demo is None:
        st.info(
            "Aucun jeu de démonstration n'est disponible pour l'instant "
            "(`module2_demo_data.csv` manquant) — utilise le mode import."
        )
    else:
        names = bearing_display_names(demo["bearing"].unique())
        bearing_choice = st.selectbox(
            "Choisir un roulement", sorted(demo["bearing"].unique()), format_func=lambda b: names[b]
        )
        sub = demo[demo.bearing == bearing_choice].sort_values("file_index").reset_index(drop=True)

        rank = st.select_slider(
            "Étape de vie du roulement",
            options=list(range(len(sub))),
            value=len(sub) // 2,
            format_func=lambda i: stage_label(i, len(sub)),
        )
        row = sub.iloc[[rank]]
        pred_rul = float(model.predict(row[FEATURE_COLS])[0])
        true_rul = float(row["RUL_pct"].iloc[0])

        c1, c2, c3 = st.columns(3)
        c1.metric("RUL prédit (%)", f"{pred_rul:.0f}")
        c2.metric("RUL réel (%)", f"{true_rul:.0f}")
        c3.metric("Health Index", f"{pred_rul:.0f} / 100", delta=health_zone(pred_rul))

        st.subheader("Évolution du RUL sur toute la vie du roulement")
        all_preds = model.predict(sub[FEATURE_COLS])
        chart_df = pd.DataFrame(
            {"RUL réel (%)": sub["RUL_pct"].values, "RUL prédit (%)": all_preds},
            index=sub["file_index"],
        )
        st.line_chart(chart_df)

        importances = safe_feature_importance(model, FEATURE_COLS)
        if importances is not None:
            st.subheader("Facteurs les plus déterminants pour cette prédiction")
            st.bar_chart(importances)
        else:
            st.caption("Ce modèle n'expose pas directement l'importance des variables.")

else:
    st.markdown(
        "Importe un fichier CSV contenant, pour chaque instantané vibratoire, les colonnes "
        f"**{', '.join(MODULE2_SIGNAL_COLS)}**, avec une colonne `snapshot_id` si le fichier "
        "couvre plusieurs instantanés successifs."
    )
    uploaded = st.file_uploader("Fichier CSV", type="csv", key="m2_upload")
    if uploaded:
        raw = pd.read_csv(uploaded)
        missing = [c for c in MODULE2_SIGNAL_COLS if c not in raw.columns]
        if missing:
            st.error(f"Colonnes manquantes dans le fichier : {missing}")
        else:
            group_col = "snapshot_id" if "snapshot_id" in raw.columns else None
            if group_col is None:
                feat = extract_features_module2(raw)
                pred_rul = float(model.predict(pd.DataFrame([feat])[FEATURE_COLS])[0])
                st.metric("RUL prédit (%)", f"{pred_rul:.0f}")
            else:
                rows = []
                for sid, g in raw.groupby(group_col):
                    feat = extract_features_module2(g)
                    feat[group_col] = sid
                    rows.append(feat)
                feats = pd.DataFrame(rows).sort_values(group_col)
                preds = model.predict(feats[FEATURE_COLS])
                feats["RUL_prédit"] = preds
                st.line_chart(feats.set_index(group_col)["RUL_prédit"])
                st.dataframe(feats[[group_col, "RUL_prédit"]])
