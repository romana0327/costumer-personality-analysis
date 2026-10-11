"""
Customer Personality Analysis — Application Streamlit
Prediction de la reponse client avec probabilite
Modele : XGBoost + SMOTE

Lancement :
    streamlit run app.py

Fichiers requis dans le meme dossier :
    modele_final.pkl
    scaler.pkl
    le_marital.pkl
    meta_modele.json
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt


# CONFIGURATION PAGE

st.set_page_config(
    page_title="Customer Personality — Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS PERSONNALISE — STYLE MODERNE ET ÉPURÉ

st.markdown("""
<style>
    /* ─── FOND GLOBAL ─── */
    .stApp {
        background: #f5f7fa;
    }
    
    /* ─── HEADER PRINCIPAL ─── */
    .main-header {
        background: linear-gradient(135deg, #1a2a4a 0%, #2a4a7a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(26, 42, 74, 0.15);
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 250px;
        height: 250px;
        background: rgba(255,255,255,0.03);
        border-radius: 50%;
    }
    .main-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .main-header p {
        font-size: 0.85rem;
        margin: 0.2rem 0 0;
        opacity: 0.8;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    .main-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        padding: 0.15rem 0.8rem;
        border-radius: 16px;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        margin-top: 0.3rem;
        position: relative;
        z-index: 1;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ─── CARTES METRIQUES ─── */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e8ecf0;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border-color: #c8d0d8;
    }
    .metric-label {
        font-size: 0.65rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a2a4a;
        margin: 0.1rem 0;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.65rem;
        color: #95a5a6;
    }

    /* ─── SECTION TITRE ─── */
    .section-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5d6d7e;
        border-left: 3px solid #2a4a7a;
        padding-left: 0.6rem;
        margin-bottom: 0.6rem;
    }

    /* ─── RESULTAT POSITIF ─── */
    .result-positive {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 1.5px solid #43a047;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(67, 160, 71, 0.1);
    }
    .result-positive .icon {
        font-size: 2.8rem;
        display: block;
    }
    .result-positive .titre {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1b5e20;
        margin: 0.2rem 0;
    }
    .result-positive .prob {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1b5e20;
        background: rgba(255,255,255,0.4);
        padding: 0.1rem 1.2rem;
        border-radius: 8px;
        display: inline-block;
    }
    .result-positive .desc {
        color: #2e7d32;
        font-size: 0.8rem;
        margin-top: 0.3rem;
        opacity: 0.8;
    }
    .result-positive .tag {
        display: inline-block;
        background: #43a047;
        color: white;
        padding: 0.15rem 1rem;
        border-radius: 16px;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ─── RESULTAT NEGATIF ─── */
    .result-negative {
        background: linear-gradient(135deg, #fbe9e7 0%, #ffcdd2 100%);
        border: 1.5px solid #e53935;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(229, 57, 53, 0.08);
    }
    .result-negative .icon {
        font-size: 2.8rem;
        display: block;
    }
    .result-negative .titre {
        font-size: 1.3rem;
        font-weight: 700;
        color: #b71c1c;
        margin: 0.2rem 0;
    }
    .result-negative .prob {
        font-size: 2.5rem;
        font-weight: 800;
        color: #b71c1c;
        background: rgba(255,255,255,0.4);
        padding: 0.1rem 1.2rem;
        border-radius: 8px;
        display: inline-block;
    }
    .result-negative .desc {
        color: #c62828;
        font-size: 0.8rem;
        margin-top: 0.3rem;
        opacity: 0.8;
    }
    .result-negative .tag {
        display: inline-block;
        background: #e53935;
        color: white;
        padding: 0.15rem 1rem;
        border-radius: 16px;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ─── SIDEBAR ─── */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e8ecf0;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #1a2a4a 0%, #2a4a7a 100%);
        color: white;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(26, 42, 74, 0.12);
        letter-spacing: 0.2px;
    }

    /* ─── BUSINESS RULE BOX ─── */
    .biz-box {
        background: #fef9e7;
        border-left: 3px solid #f39c12;
        border-radius: 0 6px 6px 0;
        padding: 0.6rem 0.8rem;
        font-size: 0.75rem;
        color: #7d6608;
        margin-bottom: 1rem;
        border: 1px solid rgba(243, 156, 18, 0.12);
        border-left-width: 3px;
    }
    .biz-box strong {
        color: #5d3a00;
    }

    /* ─── FORMULAIRES ─── */
    .stNumberInput input, .stSelectbox select, .stSlider input {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        transition: all 0.2s ease !important;
        font-size: 0.85rem !important;
    }
    .stNumberInput input:focus, .stSelectbox select:focus, .stSlider input:focus {
        border-color: #2a4a7a !important;
        box-shadow: 0 0 0 3px rgba(42, 74, 122, 0.08) !important;
    }

    /* ─── BOUTON PREDICTION ─── */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 1rem !important;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #1a2a4a 0%, #2a4a7a 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 12px rgba(26, 42, 74, 0.15) !important;
    }
    .stButton button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(26, 42, 74, 0.2) !important;
    }
    .stButton button:not([kind="primary"]) {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        color: #1a2a4a !important;
    }
    .stButton button:not([kind="primary"]):hover {
        border-color: #2a4a7a !important;
        background: #f8fafc !important;
    }

    /* ─── EXPANDER ─── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #1a2a4a !important;
        background: white !important;
        border-radius: 8px !important;
        border: 1px solid #e8ecf0 !important;
        font-size: 0.8rem !important;
        padding: 0.4rem 0.8rem !important;
    }
    .streamlit-expanderContent {
        background: white !important;
        border-radius: 0 0 8px 8px !important;
        border: 1px solid #e8ecf0 !important;
        border-top: none !important;
        padding: 0.5rem !important;
    }

    /* ─── DIVIDER ─── */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,0,0,0.06), transparent);
        margin: 1rem 0;
    }

    /* ─── FOOTER ─── */
    .footer-text {
        text-align: center;
        font-size: 0.6rem;
        color: #aab7b8;
        padding-bottom: 0.5rem;
        letter-spacing: 0.3px;
    }
    .footer-text .sep {
        color: #c8d0d8;
        margin: 0 0.4rem;
    }

    /* ─── MESSAGE D'ACCUEIL ─── */
    .welcome-box {
        background: white;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        color: #1a2a4a;
        box-shadow: 0 2px 12px rgba(0,0,0,0.02);
    }
    .welcome-box .icon-big {
        font-size: 2.8rem;
        display: block;
    }
    .welcome-box .title {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #1a2a4a;
    }
    .welcome-box .sub {
        font-size: 0.85rem;
        opacity: 0.7;
    }
    .welcome-box .crisp-box {
        font-size: 0.75rem;
        background: #f8fafc;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        display: inline-block;
        text-align: left;
        margin-top: 0.5rem;
        border: 1px solid #e8ecf0;
    }
    .welcome-box .crisp-box .step {
        display: inline-block;
        background: #1a2a4a;
        color: white;
        border-radius: 50%;
        width: 18px;
        height: 18px;
        text-align: center;
        line-height: 18px;
        font-size: 0.5rem;
        font-weight: 700;
        margin-right: 0.3rem;
    }

    /* ─── HIDE DEFAULT ELEMENTS ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 0.8rem !important; padding-bottom: 0.5rem !important; }

    /* ─── REDUIRE LES ESPACES ─── */
    div[data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
    }
    .element-container {
        margin-bottom: 0.2rem !important;
    }
    .stColumns {
        gap: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)



# CHARGEMENT DU MODELE

@st.cache_resource
def charger_modele():
    mdl = joblib.load("modele_final.pkl")
    sc = joblib.load("scaler.pkl")
    le_mar = joblib.load("le_marital.pkl")
    with open("meta_modele.json") as f:
        meta = json.load(f)
    return mdl, sc, le_mar, meta

try:
    modele, scaler, le_marital, META = charger_modele()
    MODELE_OK = True
except FileNotFoundError as e:
    MODELE_OK = False
    st.error(f"Fichier manquant : {e}. Placez les fichiers requis dans le meme dossier.")
    st.stop()



# FONCTION DE PREDICTION

def predire(annee_naissance, education, statut_marital, revenu,
            nb_enfants, nb_ados, jours_depuis_achat, montant_vins,
            visites_web, cmp1, cmp2, cmp3, cmp4, cmp5, reclamation,
            annee_inscription):

    ANNEE_REF = META["annee_reference"]
    seuil = META["seuil_optimal"]
    feat_order = META["feature_names"]
    ordre_edu = META["ordre_education"]

    age = ANNEE_REF - annee_naissance
    anciennete = (ANNEE_REF - annee_inscription) * 365
    total_enfants = nb_enfants + nb_ados
    total_cmp = cmp1 + cmp2 + cmp3 + cmp4 + cmp5

    edu_enc = ordre_edu[education]
    mar_enc = int(le_marital.transform([statut_marital])[0])

    client = {
        "Education": edu_enc,
        "Marital_Status": mar_enc,
        "Income": revenu,
        "Kidhome": nb_enfants,
        "Teenhome": nb_ados,
        "Recency": jours_depuis_achat,
        "MntWines": montant_vins,
        "NumWebVisitsMonth": visites_web,
        "AcceptedCmp3": cmp3,
        "AcceptedCmp4": cmp4,
        "AcceptedCmp5": cmp5,
        "AcceptedCmp1": cmp1,
        "AcceptedCmp2": cmp2,
        "Complain": reclamation,
        "Age": age,
        "Anciennete_jours": anciennete,
        "Total_enfants": total_enfants,
        "Total_campagnes_acceptees": total_cmp,
    }

    X = pd.DataFrame([client])[feat_order]
    X_sc = scaler.transform(X)
    proba = float(modele.predict_proba(X_sc)[0, 1])
    decision = "OUI" if proba >= seuil else "NON"

    if proba >= 0.75:
        niveau = "Élevé"
    elif proba >= seuil:
        niveau = "Modéré"
    elif proba >= 0.30:
        niveau = "Faible"
    else:
        niveau = "Très faible"

    return {
        "proba": proba,
        "proba_pct": round(proba * 100, 1),
        "decision": decision,
        "niveau": niveau,
        "age": age,
        "anciennete": round(anciennete / 365, 1),
        "total_enfants": total_enfants,
        "score_cmp": total_cmp,
        "seuil_pct": round(seuil * 100, 1),
    }



# HEADER PRINCIPAL

st.markdown("""
<div class="main-header">
    <h1>Customer Personality Analysis</h1>
    <p>Prédiction de la réponse client — Modèle XGBoost + SMOTE</p>
    <span class="badge">Pipeline CRISP-DM · Production Ready</span>
</div>
""", unsafe_allow_html=True)

# Metriques du modele
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="metric-card">
        <div class="metric-label">Modèle</div>
        <div class="metric-value">XGBoost</div>
        <div class="metric-sub">Gradient Boosting</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">AUC-ROC</div>
        <div class="metric-value">{META['auc_roc']:.4f}</div>
        <div class="metric-sub">Performance globale</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">F1-Score</div>
        <div class="metric-value">{META['f1_score']:.4f}</div>
        <div class="metric-sub">Classe répondant</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Seuil optimal</div>
        <div class="metric-value">{META['seuil_optimal']*100:.0f}%</div>
        <div class="metric-sub">Optimisé par F1-Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)


# SIDEBAR — SAISIE DU CLIENT

with st.sidebar:
    st.markdown('<div class="sidebar-header">Saisie du client</div>', unsafe_allow_html=True)

    st.markdown('<div class="biz-box"><strong>Rappel métier</strong><br>Contact = 3 EUR · Réponse = 11 EUR<br>FN = −8 EUR · FP = −3 EUR<br>Seuil à 43% (favorise le Recall)</div>', unsafe_allow_html=True)

    # Profil Demographique
    st.markdown('<div class="section-title">Profil démographique</div>', unsafe_allow_html=True)

    annee_naissance = st.number_input(
        "Année de naissance",
        min_value=1924, max_value=1996, value=1980, step=1
    )
    education = st.selectbox(
        "Niveau d'éducation",
        options=["Basic", "2n Cycle", "Graduation", "Master", "PhD"],
        index=2
    )
    statut_marital = st.selectbox(
        "Statut marital",
        options=["Divorced", "Married", "Single", "Widow"],
        index=1
    )
    revenu = st.number_input(
        "Revenu annuel (EUR)",
        min_value=1000, max_value=200000, value=52000, step=1000
    )

    col_k, col_t = st.columns(2)
    with col_k:
        nb_enfants = st.selectbox("Enfants", [0, 1, 2], index=0)
    with col_t:
        nb_ados = st.selectbox("Ados", [0, 1, 2], index=0)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Comportement Achat
    st.markdown('<div class="section-title">Comportement d\'achat</div>', unsafe_allow_html=True)

    annee_inscription = st.number_input(
        "Année d'inscription",
        min_value=2009, max_value=2014, value=2012, step=1
    )
    jours_depuis_achat = st.slider(
        "Jours depuis dernier achat",
        min_value=0, max_value=99, value=30
    )
    montant_vins = st.number_input(
        "Dépenses vins (24 mois, EUR)",
        min_value=0, max_value=1500, value=200, step=10
    )
    visites_web = st.slider(
        "Visites web par mois",
        min_value=0, max_value=20, value=5
    )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Historique Campagnes
    st.markdown('<div class="section-title">Historique des campagnes</div>', unsafe_allow_html=True)

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        cmp1 = int(st.checkbox("Camp. 1", value=False))
        cmp4 = int(st.checkbox("Camp. 4", value=False))
    with col_c2:
        cmp2 = int(st.checkbox("Camp. 2", value=False))
        cmp5 = int(st.checkbox("Camp. 5", value=False))
    with col_c3:
        cmp3 = int(st.checkbox("Camp. 3", value=False))

    reclamation = int(st.checkbox("A fait une réclamation", value=False))

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Bouton Prediction
    btn_predict = st.button(
        "Lancer la prédiction",
        use_container_width=True,
        type="primary"
    )

    st.markdown('<div class="section-title">Exemples rapides</div>', unsafe_allow_html=True)

    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        ex1 = st.button("Profil fort", use_container_width=True)
    with ex_col2:
        ex2 = st.button("Profil faible", use_container_width=True)



# GESTION DES EXEMPLES

if ex1:
    st.session_state["exemple"] = {
        "annee_naissance": 1972, "education": "PhD",
        "statut_marital": "Married", "revenu": 76000,
        "nb_enfants": 0, "nb_ados": 1,
        "jours_depuis_achat": 10, "montant_vins": 720,
        "visites_web": 9, "cmp1": 1, "cmp2": 0, "cmp3": 1,
        "cmp4": 0, "cmp5": 1, "reclamation": 0,
        "annee_inscription": 2010,
    }
    st.rerun()

if ex2:
    st.session_state["exemple"] = {
        "annee_naissance": 1992, "education": "Basic",
        "statut_marital": "Single", "revenu": 16500,
        "nb_enfants": 2, "nb_ados": 0,
        "jours_depuis_achat": 97, "montant_vins": 8,
        "visites_web": 2, "cmp1": 0, "cmp2": 0, "cmp3": 0,
        "cmp4": 0, "cmp5": 0, "reclamation": 1,
        "annee_inscription": 2013,
    }
    st.rerun()

if "exemple" in st.session_state:
    e = st.session_state["exemple"]
    annee_naissance = e["annee_naissance"]
    education = e["education"]
    statut_marital = e["statut_marital"]
    revenu = e["revenu"]
    nb_enfants = e["nb_enfants"]
    nb_ados = e["nb_ados"]
    jours_depuis_achat = e["jours_depuis_achat"]
    montant_vins = e["montant_vins"]
    visites_web = e["visites_web"]
    cmp1 = e["cmp1"]
    cmp2 = e["cmp2"]
    cmp3 = e["cmp3"]
    cmp4 = e["cmp4"]
    cmp5 = e["cmp5"]
    reclamation = e["reclamation"]
    annee_inscription = e["annee_inscription"]
    btn_predict = True
    del st.session_state["exemple"]



# ZONE PRINCIPALE — RESULTATS

age_calcule = META["annee_reference"] - annee_naissance
if age_calcule < 18 or age_calcule > 90:
    st.warning(f"Âge calculé : {age_calcule} ans — hors plage métier [18-90 ans].")

if btn_predict:
    with st.spinner("Calcul en cours..."):
        res = predire(
            annee_naissance, education, statut_marital, revenu,
            nb_enfants, nb_ados, jours_depuis_achat, montant_vins,
            visites_web, cmp1, cmp2, cmp3, cmp4, cmp5, reclamation,
            annee_inscription
        )

    col_res, col_viz = st.columns([1, 1], gap="medium")

    # Colonne gauche : Resultat principal
    with col_res:
        st.markdown('<div class="section-title">Résultat de la prédiction</div>', unsafe_allow_html=True)

        if res["decision"] == "OUI":
            st.markdown(f"""
            <div class="result-positive">
                <span class="icon">✓</span>
                <div class="titre">Ce client va répondre</div>
                <div class="prob">{res['proba_pct']}%</div>
                <div class="desc">Probabilité de réponse positive<br>Seuil : {res['seuil_pct']}% — <strong>Seuil dépassé</strong></div>
                <span class="tag">Cible prioritaire</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <span class="icon">✗</span>
                <div class="titre">Ce client ne répondra pas</div>
                <div class="prob">{res['proba_pct']}%</div>
                <div class="desc">Probabilité de réponse positive<br>Seuil : {res['seuil_pct']}% — <strong>Seuil non atteint</strong></div>
                <span class="tag">À ignorer</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Details calcules
        st.markdown('<div class="section-title">Détails calculés</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Âge", f"{res['age']} ans")
        c2.metric("Ancienneté", f"{res['anciennete']} ans")
        c3.metric("Enfants", str(res['total_enfants']))

        c4, c5, c6 = st.columns(3)
        c4.metric("Engagement", f"{res['score_cmp']}/5")
        c5.metric("Confiance", res['niveau'])
        c6.metric("Modèle", "XGBoost")

        st.markdown("<br>", unsafe_allow_html=True)

        # Recapitulatif
        with st.expander("Récapitulatif du profil saisi"):
            recap = pd.DataFrame({
                "Variable": [
                    "Année naissance", "Âge calculé", "Éducation",
                    "Statut marital", "Revenu", "Enfants", "Ados",
                    "Jours depuis achat", "Dépenses vins", "Visites web",
                    "Campagnes acceptées", "Réclamation", "Année inscription"
                ],
                "Valeur": [
                    annee_naissance, f"{res['age']} ans", education,
                    statut_marital, f"{revenu:,} EUR",
                    nb_enfants, nb_ados,
                    f"{jours_depuis_achat} jours",
                    f"{montant_vins} EUR",
                    visites_web,
                    f"{cmp1+cmp2+cmp3+cmp4+cmp5}/5",
                    "Oui" if reclamation else "Non",
                    annee_inscription
                ]
            })
            st.dataframe(recap, hide_index=True, use_container_width=True)

    # Colonne droite : Visualisations
    with col_viz:
        st.markdown('<div class="section-title">Visualisations</div>', unsafe_allow_html=True)

        # Jauge de probabilite
        fig, ax = plt.subplots(figsize=(6, 2.8))
        fig.patch.set_facecolor("#f8f9fa")
        ax.set_facecolor("#f8f9fa")

        proba = res["proba_pct"]
        seuil_pct = res["seuil_pct"]
        color_fill = "#43a047" if proba >= seuil_pct else "#e53935"

        ax.barh(0, 100, height=0.5, color="#e8ecf0", alpha=0.6, zorder=1)
        ax.barh(0, proba, height=0.5, color=color_fill, alpha=0.85, zorder=2)
        ax.axvline(x=seuil_pct, color="#1a2a4a", lw=2, linestyle="--", zorder=3, label=f"Seuil = {seuil_pct}%")

        ax.text(proba / 2, 0, f"{proba}%", va="center", ha="center", fontsize=16, fontweight="bold", color="white", zorder=4)
        ax.text(seuil_pct + 1, -0.3, f"Seuil\n{seuil_pct}%", fontsize=7, color="#2c3e50")

        ax.set_xlim(0, 105)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Probabilité de réponse (%)", fontsize=9)
        ax.set_title("Jauge de probabilité", fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=8, loc="upper right")
        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Radar comparatif
        st.markdown("<br>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        fig2.patch.set_facecolor("#f8f9fa")
        ax2.set_facecolor("#f8f9fa")

        vars_comp = ["Revenu\n(/1000)", "Vins\n(/10)", "Âge", "Recency", "Campagnes\n(×10)"]
        val_client = [
            revenu / 1000,
            montant_vins / 10,
            res["age"],
            jours_depuis_achat,
            (cmp1+cmp2+cmp3+cmp4+cmp5) * 10
        ]
        val_moy_0 = [52.2, 30.0, 45.2, 49.1, 2.0]
        val_moy_1 = [61.8, 145.0, 48.7, 47.5, 18.0]

        x = np.arange(len(vars_comp))
        width = 0.26

        ax2.bar(x - width, val_moy_0, width, label="Moy. non-répondants", color="#aec6cf", edgecolor="white", alpha=0.8)
        ax2.bar(x, val_moy_1, width, label="Moy. répondants", color="#66bb6a", edgecolor="white", alpha=0.8)
        ax2.bar(x + width, val_client, width, label="Ce client", color="#e53935" if res["decision"]=="NON" else "#1a2a4a", edgecolor="white", alpha=0.9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(vars_comp, fontsize=8)
        ax2.set_title("Profil client vs moyennes", fontsize=10, fontweight="bold", pad=8)
        ax2.legend(fontsize=7.5, loc="upper right")
        ax2.set_ylabel("Valeur normalisée", fontsize=8)
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)

        st.pyplot(fig2, use_container_width=True)
        plt.close()

else:
    # Message d'accueil
    st.markdown("""
    <div class="welcome-box">
        <span class="icon-big">📊</span>
        <div class="title">Prêt à prédire</div>
        <div class="sub">Remplissez le formulaire dans la barre latérale<br>puis cliquez sur <strong>Lancer la prédiction</strong></div>
        <br>
        <div class="crisp-box">
            <strong>Processus CRISP-DM</strong><br>
            <span class="step">1</span> Business Understanding<br>
            <span class="step">2</span> Data Understanding<br>
            <span class="step">3</span> Data Preparation<br>
            <span class="step">4</span> Modeling — SMOTE + XGBoost<br>
            <span class="step">5</span> Evaluation — AUC-ROC = 0.8884<br>
            <span class="step">6</span> Deployment — <strong>Vous êtes ici</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)



# FOOTER

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-text">
    Customer Personality Analysis <span class="sep">·</span>
    XGBoost + SMOTE <span class="sep">·</span>
    CRISP-DM Pipeline <span class="sep">·</span>
    AUC-ROC = 0.8884 <span class="sep">·</span>
    Seuil = 43%
</div>
""", unsafe_allow_html=True)