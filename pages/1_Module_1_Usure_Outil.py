import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pipeline_utils import (
    MODULE1_SIGNAL_COLS, extract_features_module1, health_zone,
    display_name_for_tool, stage_label, safe_feature_importance,
    inject_base_css, render_gauge_html,
)

st.set_page_config(page_title="Module 1 — Usure d'outil", page_icon="🔧", layout="wide")
inject_base_css()
st.title("🔧 Module 1 — RUL Usure d'outil")
st.caption("Prédiction de la durée de vie restante d'un outil de coupe sur fraiseuse CNC 3 axes")

with st.expander("ℹ️ Comment utiliser ce module"):
    st.markdown(
        "- **Outil A / B / C** représentent trois outils de coupe suivis jusqu'à leur fin de "
        "vie réelle, utilisés ici pour démontrer la fiabilité du modèle sur des cas connus.\n"
        "- Déplace le curseur **Étape de vie** pour voir comment le diagnostic évolue entre "
        "le début et la fin de vie de l'outil.\n"
        "- En conditions réelles, ces courbes proviendraient du flux de capteurs de ta propre "
        "machine plutôt que d'un scénario pré-enregistré."
    )
with st.expander("🔒 Détails techniques (méthodologie)"):
    st.markdown(
        "Modèle entraîné et validé (Leave-One-Out) sur des relevés réels de force, vibration "
        "et émission acoustique capturés lors d'essais de fraisage jusqu'à rupture d'outil. "
        "49 indicateurs statistiques extraits par passe de coupe, Random Forest Regressor."
    )


@st.cache_resource
def load_model():
    model = joblib.load("data/module1_model.pkl")
    feature_cols = joblib.load("data/module1_features.pkl")
    return model, feature_cols


@st.cache_data
def load_demo_data():
    return pd.read_csv("data/module1_demo_data.csv")


model, FEATURE_COLS = load_model()
demo = load_demo_data()

mode = st.radio("Mode", ["Démonstration", "Importer mes données"], horizontal=True)

if mode == "Démonstration":
    tool_options = sorted(demo["experiment_tag"].unique())
    tool_choice = st.selectbox(
        "Choisir un outil", tool_options, format_func=display_name_for_tool
    )
    sub = demo[demo.experiment_tag == tool_choice].sort_values("cut_id").reset_index(drop=True)

    rank = st.select_slider(
        "Étape de vie de l'outil",
        options=list(range(len(sub))),
        value=len(sub) // 2,
        format_func=lambda i: stage_label(i, len(sub)),
    )
    row = sub.iloc[[rank]]
    pred_rul = float(model.predict(row[FEATURE_COLS])[0])
    true_rul = float(row["RUL"].iloc[0])
    health = float(np.clip(100 * pred_rul / sub["RUL"].max(), 0, 100))

    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.metric("RUL prédit (passes restantes)", f"{pred_rul:.0f}")
    c2.metric("RUL réel", f"{true_rul:.0f}")
    with c3:
        st.markdown(render_gauge_html(health, health_zone(health)), unsafe_allow_html=True)

    st.subheader("Évolution du RUL sur toute la vie de l'outil")
    all_preds = model.predict(sub[FEATURE_COLS])
    chart_df = pd.DataFrame(
        {"RUL réel": sub["RUL"].values, "RUL prédit": all_preds}, index=sub["cut_id"]
    )
    st.line_chart(chart_df)

    importances = safe_feature_importance(model, FEATURE_COLS)
    if importances is not None:
        st.subheader("Facteurs les plus déterminants pour cette prédiction")
        st.bar_chart(importances)

else:
    st.markdown(
        "Importe un fichier CSV contenant, pour chaque passe de coupe, les colonnes suivantes "
        "(une ligne par échantillon temporel) : `cut_id` (ou équivalent), `tool_wear`, et "
        f"**{', '.join(MODULE1_SIGNAL_COLS)}**."
    )
    uploaded = st.file_uploader("Fichier CSV", type="csv", key="m1_upload")
    if uploaded:
        raw = pd.read_csv(uploaded)
        missing = [c for c in MODULE1_SIGNAL_COLS if c not in raw.columns]
        if missing:
            st.error(f"Colonnes manquantes dans le fichier : {missing}")
        else:
            group_col = "cut_id" if "cut_id" in raw.columns else None
            if group_col is None:
                st.warning(
                    "Pas de colonne `cut_id` détectée : le fichier entier est traité comme "
                    "une seule passe de coupe."
                )
                feat = extract_features_module1(raw)
                pred_rul = float(model.predict(pd.DataFrame([feat])[FEATURE_COLS])[0])
                st.metric("RUL prédit (passes restantes)", f"{pred_rul:.0f}")
            else:
                rows = []
                for cid, g in raw.groupby(group_col):
                    feat = extract_features_module1(g)
                    feat[group_col] = cid
                    rows.append(feat)
                feats = pd.DataFrame(rows).sort_values(group_col)
                preds = model.predict(feats[FEATURE_COLS])
                feats["RUL_prédit"] = preds
                st.line_chart(feats.set_index(group_col)["RUL_prédit"])
                st.dataframe(feats[[group_col, "RUL_prédit"]])
