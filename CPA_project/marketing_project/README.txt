╔══════════════════════════════════════════════════════════════╗
║        Customer Personality Analysis — Guide Complet         ║
╚══════════════════════════════════════════════════════════════╝

STRUCTURE DU DOSSIER :
──────────────────────
marketing_project/
│
├── marketing_campaignvrai.csv          ← Dataset original
├── Customer_Personality_COMPLET.ipynb  ← Notebook principal (CRISP-DM A-Z)
├── code_visualisation_smote.py         ← Graphiques SMOTE (copier dans notebook)
├── app.py                              ← Application Streamlit (deployment)
│
├── modele_final.pkl     ─┐
├── scaler.pkl            ├── Generes automatiquement par le notebook
├── le_marital.pkl        │   (ne pas toucher)
└── meta_modele.json     ─┘

ORDRE D EXECUTION DANS VS CODE :
─────────────────────────────────
ETAPE 1 : Ouvrir VS Code → File > Open Folder → selectionner ce dossier

ETAPE 2 : Ouvrir Customer_Personality_COMPLET.ipynb
          → Cliquer "Run All" (Ctrl+F9)
          → Les fichiers .pkl sont generes automatiquement

ETAPE 3 : Ouvrir le terminal VS Code (Ctrl + ù)
          → Installer les dependances :
          pip install streamlit xgboost scikit-learn imbalanced-learn joblib matplotlib pandas numpy

ETAPE 4 : Lancer l application :
          streamlit run app.py
          → L app s ouvre dans le navigateur sur http://localhost:8501

PHASES CRISP-DM COUVERTES :
────────────────────────────
Phase 1 — Business Understanding  : contexte, logique economique
Phase 2 — Data Understanding      : exploration, anomalies
Phase 3 — Data Preparation        : nettoyage, encodage, feature engineering
Phase 4 — Modeling                : SMOTE + 3 modeles (LR, RF, XGBoost)
Phase 5 — Evaluation              : AUC-ROC, F1, Recall, matrices confusion
Phase 6 — Deployment              : app.py (Streamlit)

PERFORMANCES DU MODELE FINAL (XGBoost) :
─────────────────────────────────────────
AUC-ROC  : 0.8884
F1-Score : 0.6222
Recall   : 0.6364
Seuil    : 43%
