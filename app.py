import streamlit as st
import os

st.set_page_config(page_title="CNC Sentinel", page_icon="🛠️", layout="wide")

# ---- Style personnalisé (au-delà du thème Streamlit de base) ----
st.markdown("""
<style>
    .block-container {padding-top: 2.5rem; max-width: 1100px;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        margin-bottom: 2rem;
    }
    .hero h1 {font-size: 2.4rem; margin-bottom: 0.3rem;}
    .hero p {font-size: 1.05rem; color: #94A3B8; max-width: 640px;}
    .badge {
        display: inline-block; background:#F97316; color:#0F172A;
        padding: 3px 12px; border-radius: 999px; font-size:0.75rem;
        font-weight:700; letter-spacing:0.03em; margin-bottom:1rem;
    }
    .pricing-card {
        border: 1px solid #334155; border-radius: 14px; padding: 1.6rem;
        background: #1E293B; height: 100%;
    }
    .pricing-card.featured {border: 1px solid #F97316;}
    .price {font-size: 2rem; font-weight: 700; margin: 0.3rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <span class="badge">DIAGNOSTIC PRÉDICTIF · CNC 3 AXES</span>
    <h1>🛠️ CNC Sentinel</h1>
    <p>
        Anticipez les pannes mécaniques de vos fraiseuses CNC avant qu'elles ne surviennent.
        Deux modules de diagnostic — usure d'outil et roulements — construits sur une
        méthodologie standardisée et validée sur des jeux de données de référence en
        maintenance prédictive (PHM).
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Modules de diagnostic")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 🔧 Usure d'outil")
        st.markdown(
            "Prédit le nombre de passes restantes avant fin de vie de l'outil de coupe, "
            "à partir des signaux de force, vibration et émission acoustique."
        )
        st.metric("Précision de validation (R²)", "0.93 – 0.95")
        st.page_link("pages/1_Module_1_Usure_Outil.py", label="Ouvrir le module →", icon="🔧")

with col2:
    with st.container(border=True):
        st.markdown("#### ⚙️ Roulements")
        st.markdown(
            "Prédit le pourcentage de vie restante d'un roulement, à partir de sa "
            "signature vibratoire (accélération horizontale et verticale)."
        )
        module2_ready = os.path.exists("data/module2_model.pkl")
        st.metric("Statut du module", "Actif ✅" if module2_ready else "Bientôt disponible")
        st.page_link("pages/2_Module_2_Roulements.py", label="Ouvrir le module →", icon="⚙️")

with st.container(border=True):
    st.markdown("#### 🩺 Vue globale — Health Index machine")
    st.markdown("Fusionne les deux modules en un score unique de santé machine, pour un diagnostic d'ensemble en un coup d'œil.")
    st.page_link("pages/3_Sante_Globale.py", label="Voir la santé globale →", icon="🩺")

st.markdown("---")
st.markdown("### Comment ça marche")
c1, c2, c3 = st.columns(3)
c1.markdown("**1. Choisissez un module**\n\nOutil de coupe ou roulement, selon le composant à surveiller.")
c2.markdown("**2. Testez ou importez**\n\nDémo instantanée avec données publiques, ou import de vos propres relevés capteurs.")
c3.markdown("**3. Obtenez un diagnostic**\n\nRUL estimé, Health Index et facteurs explicatifs, en quelques secondes.")

st.markdown("---")
st.markdown("### Offres")
st.caption("Aperçu du modèle de déploiement envisagé pour une mise en production industrielle.")

p1, p2, p3 = st.columns(3)
with p1:
    st.markdown("""
    <div class="pricing-card">
        <b>Découverte</b>
        <div class="price">Gratuit</div>
        <p style="color:#94A3B8;font-size:0.9rem;">
        • Démo avec données publiques<br>
        • 1 diagnostic à la fois<br>
        • Support communautaire
        </p>
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown("""
    <div class="pricing-card featured">
        <b>Atelier</b>
        <div class="price">Sur devis</div>
        <p style="color:#94A3B8;font-size:0.9rem;">
        • Import de données réelles machine<br>
        • Historique et suivi multi-machines<br>
        • Alertes par email
        </p>
    </div>
    """, unsafe_allow_html=True)
with p3:
    st.markdown("""
    <div class="pricing-card">
        <b>Usine</b>
        <div class="price">Sur devis</div>
        <p style="color:#94A3B8;font-size:0.9rem;">
        • Intégration multi-lignes<br>
        • API et export de rapports<br>
        • Accompagnement dédié
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Système construit et validé sur des jeux de données publics de référence "
    "(PHM2010, FEMTO-ST/PRONOSTIA) — méthodologie reproductible, transférable "
    "à toute machine-outil 3 axes instrumentée."
)
