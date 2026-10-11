# ╔══════════════════════════════════════════════════════════════════╗
# ║   VISUALISATION COMPLÈTE — AVANT / APRÈS SMOTE                  ║
# ║   Copier-coller dans une cellule du notebook                     ║
# ║   JUSTE APRÈS la cellule "APPLICATION DU SMOTE"                 ║
# ╚══════════════════════════════════════════════════════════════════╝

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Essayer d'utiliser les variables du notebook
try:
    # On vérifie la présence des variables clés du notebook
    y_train
    y_train_sm
    X_train_sc
    X_train_sm
    FEATURE_NAMES
    
    y_train_arr = np.asarray(y_train)
    y_train_sm_arr = np.asarray(y_train_sm)
    X_train_sc_arr = np.asarray(X_train_sc)
    X_train_sm_arr = np.asarray(X_train_sm)
    feature_names_list = list(FEATURE_NAMES)
    
    print("[INFO] Variables du notebook détectées en mémoire.")
    
    # Vérification des modèles
    try:
        lr; rf; xgb; s_lr; s_rf; s_xgb; X_test_sc; y_test
        y_test_arr = np.asarray(y_test)
        X_test_sc_arr = np.asarray(X_test_sc)
        modeles_disponibles = True
        print("[INFO] Modèles et seuils détectés en mémoire.")
    except NameError:
        modeles_disponibles = False
        print("[INFO] Les modèles ne sont pas encore entraînés dans le notebook (le Graphique 4 sera masqué).")
        
except NameError:
    # Mode autonome : chargement et traitement automatique pour éviter tout NameError
    print("[INFO] Mode autonome détecté (ou variables du notebook non détectées).")
    print("[INFO] Reconstruction du pipeline de données et entraînement des modèles en cours...")
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score
    
    try:
        # Chargement
        df = pd.read_csv("marketing_campaignvrai.csv")
        
        # Preprocessing (Phases 3 & 4)
        df["Age"] = 2014 - df["Year_Birth"]
        df = df[(df["Age"] >= 18) & (df["Age"] <= 90)]
        
        borne_sup = df["Income"].quantile(0.75) + 3 * (df["Income"].quantile(0.75) - df["Income"].quantile(0.25))
        df = df[df["Income"] <= borne_sup]
        df["Income"] = df["Income"].fillna(df["Income"].median())
        
        df = df[~df["Marital_Status"].isin(["Absurd", "YOLO"])]
        df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single", "Together": "Married"})
        
        df = df.drop(columns=["NumWebVisitsMonth"])
        df = df.rename(columns={"NumWebVisitsMonth.1": "NumWebVisitsMonth"})
        df = df.drop(columns=["Z_CostContact", "Z_Revenue", "Year_Birth", "ID"])
        
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, format="mixed")
        date_ref = df["Dt_Customer"].max() + pd.Timedelta(days=1)
        df["Anciennete_jours"] = (date_ref - df["Dt_Customer"]).dt.days
        df = df.drop(columns=["Dt_Customer"])
        
        df["Total_enfants"] = df["Kidhome"] + df["Teenhome"]
        df["Total_campagnes_acceptees"] = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]].sum(axis=1)
        
        df["Education"] = df["Education"].map({"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4})
        
        le_marital = LabelEncoder()
        df["Marital_Status"] = le_marital.fit_transform(df["Marital_Status"])
        
        # Split & Scale & SMOTE
        X = df.drop(columns=["Response"])
        y = df["Response"]
        FEATURE_NAMES = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
        
        # Modèles & Seuils
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_train_sm, y_train_sm)
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(X_train_sm, y_train_sm)
        
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", verbosity=0)
        xgb.fit(X_train_sm, y_train_sm)
        
        def trouver_seuil_optimal(modele, X_te, y_te):
            prob = modele.predict_proba(X_te)[:, 1]
            seuils = np.arange(0.10, 0.80, 0.01)
            f1s = [f1_score(y_te, (prob >= s).astype(int), zero_division=0) for s in seuils]
            return round(float(seuils[np.argmax(f1s)]), 2)
            
        s_lr = trouver_seuil_optimal(lr, X_test_sc, y_test)
        s_rf = trouver_seuil_optimal(rf, X_test_sc, y_test)
        s_xgb = trouver_seuil_optimal(xgb, X_test_sc, y_test)
        
        # Passage en variables locales robustes
        y_train_arr = np.asarray(y_train)
        y_train_sm_arr = np.asarray(y_train_sm)
        X_train_sc_arr = np.asarray(X_train_sc)
        X_train_sm_arr = np.asarray(X_train_sm)
        feature_names_list = list(FEATURE_NAMES)
        
        y_test_arr = np.asarray(y_test)
        X_test_sc_arr = np.asarray(X_test_sc)
        modeles_disponibles = True
        print("[INFO] Modèles entraînés et variables initialisées avec succès.")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger ou de traiter les données : {e}")
        print("Assurez-vous que 'marketing_campaignvrai.csv' est présent dans le même dossier.")
        raise e

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPHIQUE 1 — Distribution des classes avant vs après SMOTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
counts_avant = np.bincount(y_train_arr.astype(int))
counts_apres = np.bincount(y_train_sm_arr.astype(int))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

x = np.arange(2)
width = 0.35
bars1 = axes[0].bar(x - width/2, counts_avant, width,
                    color=["#5b9bd5","#d62728"], edgecolor="white")
bars2 = axes[0].bar(x + width/2, counts_apres, width,
                    color=["#aed6f1","#f1948a"], edgecolor="white", alpha=0.85)

for bar, v in zip(list(bars1)+list(bars2), list(counts_avant)+list(counts_apres)):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+15, str(v),
                 ha="center", fontweight="bold", fontsize=10)

axes[0].set_xticks(x)
axes[0].set_xticklabels(["Classe 0\n(non-repondant)", "Classe 1\n(repondant)"])
axes[0].set_ylabel("Nombre d exemples")
axes[0].set_title("Distribution des classes\nAVANT vs APRES SMOTE", fontsize=12)

# Légende corrigée et explicite
handles = [bars1[0], bars1[1], bars2[0], bars2[1]]
labels = ["Cl.0 Avant SMOTE", "Cl.1 Avant SMOTE", "Cl.0 Après SMOTE", "Cl.1 Après SMOTE"]
axes[0].legend(handles, labels, ncol=2, fontsize=9)

for ax_i, (counts, titre) in enumerate([(counts_avant,"AVANT SMOTE\n(desequilibre)"),(counts_apres,"APRES SMOTE\n(equilibre)")], 1):
    axes[ax_i].pie(counts, labels=[f"Cl.0 : {counts[0]:,}",f"Cl.1 : {counts[1]:,}"],
                   colors=["#5b9bd5","#d62728"], autopct="%1.1f%%", startangle=90,
                   wedgeprops={"edgecolor":"white","linewidth":2.5})
    axes[ax_i].set_title(titre, fontsize=12, fontweight="bold")

plt.suptitle("Effet du SMOTE sur la Distribution des Classes", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"AVANT : Cl.0={counts_avant[0]:,} | Cl.1={counts_avant[1]:,} | Ratio={counts_avant[0]/counts_avant[1]:.1f}:1")
print(f"APRES : Cl.0={counts_apres[0]:,} | Cl.1={counts_apres[1]:,} | Ratio={counts_apres[0]/counts_apres[1]:.1f}:1")
print(f"Exemples synthetiques generes : {counts_apres[1]-counts_avant[1]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPHIQUE 2 — Distribution de 3 features clés avant vs après
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
idx_income = feature_names_list.index("Income")
idx_age    = feature_names_list.index("Age")
idx_wines  = feature_names_list.index("MntWines")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
infos = [(idx_income,"Income (normalise)","#3498db"),
         (idx_age,   "Age (normalise)",   "#2ecc71"),
         (idx_wines, "MntWines (normalise)","#e67e22")]

for col, (idx, label, color) in enumerate(infos):
    for cls, c, a, lbl in [(0,"#aec6cf",0.7,"Classe 0"),(1,"#d62728",0.85,"Classe 1")]:
        mask = y_train_arr == cls
        axes[0,col].hist(X_train_sc_arr[mask,idx], bins=30, color=c, alpha=a,
                         label=lbl, edgecolor="white", density=True)
    axes[0,col].set_title(f"{label}\nAVANT SMOTE", fontsize=11)
    axes[0,col].set_ylabel("Densite"); axes[0,col].legend(fontsize=9)

    for cls, c, a, lbl in [(0,"#aec6cf",0.7,"Classe 0 (reel)"),(1,"#d62728",0.85,"Classe 1 (reel + synthetique)")]:
        mask = y_train_sm_arr == cls
        axes[1,col].hist(X_train_sm_arr[mask,idx], bins=30, color=c, alpha=a,
                         label=lbl, edgecolor="white", density=True)
    axes[1,col].set_title(f"{label}\nAPRES SMOTE", fontsize=11)
    axes[1,col].set_ylabel("Densite"); axes[1,col].legend(fontsize=9)

plt.suptitle("Distribution des Features — AVANT vs APRES SMOTE", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPHIQUE 3 — Projection PCA 2D avant vs après SMOTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pca_av = PCA(n_components=2, random_state=42)
X_2d_av = pca_av.fit_transform(X_train_sc_arr)
pca_ap = PCA(n_components=2, random_state=42)
X_2d_ap = pca_ap.fit_transform(X_train_sm_arr)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
couleurs = {0:"#aec6cf", 1:"#d62728"}

for cls in [0,1]:
    mask = y_train_arr == cls
    axes[0].scatter(X_2d_av[mask,0], X_2d_av[mask,1], c=couleurs[cls],
                    s=12 if cls==0 else 25, alpha=0.5, edgecolors="none",
                    label=f"Classe {cls} ({mask.sum()})")
axes[0].set_title(f"AVANT SMOTE — {counts_avant[1]} exemples classe 1", fontsize=11)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=9)

for cls in [0,1]:
    mask = y_train_sm_arr == cls
    if cls == 1:
        n_reels = counts_avant[1]
        idx_cls1 = np.where(y_train_sm_arr==1)[0]
        axes[1].scatter(X_2d_ap[idx_cls1[:n_reels],0], X_2d_ap[idx_cls1[:n_reels],1],
                        c="#d62728", s=30, alpha=0.8, edgecolors="none",
                        label=f"Classe 1 reelle ({n_reels})")
        axes[1].scatter(X_2d_ap[idx_cls1[n_reels:],0], X_2d_ap[idx_cls1[n_reels:],1],
                        c="#ff7f0e", s=15, alpha=0.5, edgecolors="none",
                        label=f"Classe 1 synthetique ({len(idx_cls1)-n_reels})")
    else:
        axes[1].scatter(X_2d_ap[mask,0], X_2d_ap[mask,1], c="#aec6cf",
                        s=8, alpha=0.35, edgecolors="none", label=f"Classe 0 ({mask.sum()})")
axes[1].set_title(f"APRES SMOTE — Orange = exemples synthetiques", fontsize=11)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend(fontsize=9)

plt.suptitle("PCA 2D — Visualisation de l effet du SMOTE", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPHIQUE 4 — Probabilités prédites par les 3 modèles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if modeles_disponibles:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    modeles_viz = [(lr,"Logistic Regression","#3498db",s_lr),
                   (rf,"Random Forest",      "#2ecc71",s_rf),
                   (xgb,"XGBoost",           "#e67e22",s_xgb)]

    for ax, (mdl, nom, color, seuil) in zip(axes, modeles_viz):
        y_prob = mdl.predict_proba(X_test_sc_arr)[:,1]
        ax.hist(y_prob[y_test_arr==0], bins=30, alpha=0.70, color="#aec6cf",
                edgecolor="white", density=True, label=f"Reel : Classe 0 (n={(y_test_arr==0).sum()})")
        ax.hist(y_prob[y_test_arr==1], bins=30, alpha=0.85, color="#d62728",
                edgecolor="white", density=True, label=f"Reel : Classe 1 (n={(y_test_arr==1).sum()})")
        ax.axvline(x=seuil, color="black", lw=2.5, ls="--", label=f"Seuil = {seuil}")
        ax.axvline(x=0.50,  color="gray",  lw=1.5, ls=":", alpha=0.6, label="0.5 (defaut)")
        ax.set_xlabel("Probabilite predite"); ax.set_ylabel("Densite")
        ax.set_title(f"{nom}\n(seuil optimal = {seuil})", fontsize=11)
        ax.legend(fontsize=8.5); ax.set_xlim(0,1)

    plt.suptitle("Distribution des Probabilites Predites — 3 Modeles vs Test Set",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()
    print("Bleu (cl.0) a gauche + Rouge (cl.1) a droite = bon modele.")
else:
    print("\n[INFO] Graphique 4 (Probabilités prédites) non affiché car les modèles (lr, rf, xgb)")
    print("ne sont pas encore entraînés. Réexécutez cette cellule à la fin du notebook (après la Phase 5)")
    print("pour visualiser le Graphique 4.")
