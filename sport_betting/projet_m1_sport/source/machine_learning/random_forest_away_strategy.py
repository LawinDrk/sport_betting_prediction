import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from machine_learning.fonctions_utilitaires import *
import os
import pickle


def simulation_ai_with_strategy(df_original, capital_initial=100, mise=1):
    """
    Simule une stratégie de paris sportifs basée sur l'intelligence artificielle (Random Forest).
    Cible spécifiquement les équipes visiteuses (Away) ayant une cote comprise entre 3.0 et 4.0.
    Utilise un apprentissage incrémentiel (walk-forward) avec réentraînement saisonnier régulier.
    
    Paramètres:
    - df_original: DataFrame historique complet
    - capital_initial: Solde initial disponible pour miser
    - mise: Montant fixe misé sur chaque pari éligible
    """
    # Préparation des variables et encodages
    data, le_team, le_res = preparer_donnees_ml(df_original)

    features = ['HomeTeam_Code', 'AwayTeam_Code', 'B365H', 'B365D', 'B365A']
    X = data[features]
    y = data['FTR_Code']
    
    # Seuil d'historique initial de 100 matchs pour permettre au modèle de s'entraîner correctement au départ
    start_idx = 100 
    
    # Initialisation des variables de simulation
    capital = capital_initial
    historique_capital = [capital]
    nb_paris = 0
    nb_victoires = 0
    
    # Modèle Random Forest configuré avec les hyperparamètres optimisés par Grid Search
    model = RandomForestClassifier(
        n_estimators=100,         # 100 arbres de décision pour une bonne robustesse
        max_depth=None,           # Pas de limite de profondeur pour capturer la complexité
        min_samples_split=10,     # Au moins 10 échantillons pour séparer un nœud interne (évite le surapprentissage)
        class_weight='balanced',  # Ajustement du poids des classes pour gérer les déséquilibres (les victoires à l'extérieur sont plus rares)
        random_state=42           # Graine aléatoire fixée pour assurer la reproductibilité des résultats
    )
    
    # Récupération du code numérique correspondant à une victoire à l'extérieur ('A')
    try:
        away_code = list(le_res.classes_).index('A')
    except ValueError:
        away_code = -1
    
    # Entraînement initial sur la première tranche de l'historique
    model.fit(X.iloc[:start_idx], y.iloc[:start_idx])
    
    # Boucle de simulation temporelle, match par match
    for i in range(start_idx, len(data)):
        if capital < mise:
            break  # Arrêt si le capital est insuffisant pour placer une nouvelle mise
            
        # Apprentissage incrémentiel : réentraînement régulier tous les 100 matchs
        # Cela permet au modèle de s'adapter aux dynamiques de la saison en cours (mises à jour de forme, transferts...)
        if i % 100 == 0:
            model.fit(X.iloc[:i], y.iloc[:i])
            
        row = data.iloc[i]
        cote_away = row['B365A']
        
        # Filtre heuristique de base : la cote de l'équipe extérieure doit se situer dans la zone Value [3.0 - 4.0]
        if 3.0 <= cote_away <= 4.0:
            # Estimation de la probabilité de victoire par le modèle
            probs = model.predict_proba(X.iloc[[i]])[0]
            
            # Recherche de l'index interne de la classe 'A' dans le modèle
            idx_in_model = -1
            for idx_cls, cls in enumerate(model.classes_):
                if cls == away_code:
                    idx_in_model = idx_cls
                    break
            
            prob_away = probs[idx_in_model] if idx_in_model != -1 else 0.0
            
            # Prise de décision basée sur l'Espérance de Gain (Expected Value)
            # On ne parie que si l'espérance mathématique de gain est strictement supérieure à 5% (EV >= 1.05)
            if prob_away * cote_away >= 1.05:
                # Placer le pari
                capital -= mise
                nb_paris += 1
                
                # Si l'équipe extérieure gagne, on empoche le gain (mise * cote)
                if row['FTR'] == 'A':
                    capital += mise * cote_away
                    nb_victoires += 1
                
                # Mise à jour de l'évolution du capital uniquement lorsqu'un pari a été réellement effectué (évite les paliers plats)
                historique_capital.append(capital)
        
    return calcul_resultats(capital_initial, capital, nb_paris, nb_victoires, historique_capital, mise)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_rf_model.pkl")

def load_or_train_model(df_historical=None):
    """
    Gère la persistance (sérialisation) du modèle Random Forest.
    Tente de charger le modèle et les encodeurs à partir d'un fichier pickle pour un chargement instantané.
    Si le fichier est absent ou corrompu, réentraîne le modèle sur le jeu de données historiques complet et sauvegarde le cache.
    """
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                saved_data = pickle.load(f)
            print(f"[Modèle IA] Chargement du modèle pré-entraîné depuis le cache : {MODEL_PATH}")
            return saved_data["model"], saved_data["le_team"], saved_data["le_res"]
        except Exception as e:
            print(f"[Erreur Cache Modèle] Impossible de lire le fichier sauvegardé : {e}. Réentraînement...")
            
    # Entraînement à partir de zéro si aucun cache n'existe
    if df_historical is None:
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "charger_data", "epl_historique.csv")
        if os.path.exists(csv_path):
            df_historical = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Données historiques introuvables au chemin : {csv_path}")
            
    # Préparation des données historiques pour l'apprentissage
    data, le_team, le_res = preparer_donnees_ml(df_historical)
    features = ['HomeTeam_Code', 'AwayTeam_Code', 'B365H', 'B365D', 'B365A']
    X = data[features]
    y = data['FTR_Code']
    
    # Configuration et ajustement du modèle Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X, y)
    
    # Sauvegarde dans le fichier cache pkl pour les futures requêtes
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": rf_model, "le_team": le_team, "le_res": le_res}, f)
        print(f"[Modèle IA] Modèle entraîné et sauvegardé avec succès dans : {MODEL_PATH}")
    except Exception as e:
        print(f"[Erreur Sauvegarde Modèle] Impossible d'écrire le cache : {e}")
        
    return rf_model, le_team, le_res

if __name__ == "__main__":
    # Script de test local pour s'assurer du bon fonctionnement de la stratégie
    try:
        path = os.path.join(os.path.dirname(__file__), '../charger_data/epl_historique.csv')
        df = pd.read_csv(path)
        res = simulation_ai_with_strategy(df)
        print(f"--- Stratégie RF Extérieur - Validation ---")
        print(f"ROI : {res['roi']}%")
        print(f"Profit : {res['profit']}€")
        print(f"Nombre de paris : {res['nb_paris']}")
        print(f"Taux de victoire : {res['win_rate']}%")
    except Exception as e:
        print(f"Erreur lors du test : {e}")
