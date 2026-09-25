import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pipeline_utils import MODULE1_SIGNAL_COLS, extract_features_module1, health_zone

st.set_page_config(page_title="Module 1 — Usure d'outil", page_icon="🔧", layout="wide")
st.title("🔧 Module 1 — RUL Usure d'outil")
st.caption("Dataset : PHM2010 Milling Data Challenge — fraiseuse CNC 3 axes")


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

mode = st.radio(
    "Mode", ["Démo (données publiques PHM2010)", "Importer mes données"], horizontal=True
)

if mode.startswith("Démo"):
    cutter = st.selectbox("Choisir un outil", sorted(demo["experiment_tag"].unique()))
    sub = demo[demo.experiment_tag == cutter].sort_values("cut_id").reset_index(drop=True)
    max_cut = int(sub["cut_id"].max())

    cut_choice = st.slider(
        "Position dans la vie de l'outil (n° de la passe de coupe)", 1, max_cut, max_cut // 2
    )
    row = sub[sub.cut_id == cut_choice].iloc[[0]]
    pred_rul = float(model.predict(row[FEATURE_COLS])[0])
    true_rul = float(row["RUL"])
    health = float(np.clip(100 * pred_rul / sub["RUL"].max(), 0, 100))

    c1, c2, c3 = st.columns(3)
    c1.metric("RUL prédit (passes restantes)", f"{pred_rul:.0f}")
    c2.metric("RUL réel", f"{true_rul:.0f}")
    c3.metric("Health Index", f"{health:.0f} / 100", delta=health_zone(health))

    st.subheader("Évolution du RUL sur toute la vie de l'outil")
    all_preds = model.predict(sub[FEATURE_COLS])
    chart_df = pd.DataFrame(
        {"RUL réel": sub["RUL"].values, "RUL prédit": all_preds}, index=sub["cut_id"]
    )
    st.line_chart(chart_df)

    st.subheader("Facteurs les plus déterminants pour cette prédiction")
    importances = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(importances)

else:
    st.markdown(
        "Importe un fichier CSV contenant, pour chaque passe de coupe, les colonnes suivantes "
        "(une ligne par échantillon temporel) : `experiment_tag`, `cut_id` (ou équivalent), "
        f"`tool_wear`, et **{', '.join(MODULE1_SIGNAL_COLS)}**."
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
