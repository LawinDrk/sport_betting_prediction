import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from machine_learning.fonctions_utilitaires import *


def simulation_ml_strategy(df_original, capital_initial=100, mise=1, n_estimators=100, max_depth=5, random_state=42, seuil_confiance=0, use_kelly=False, kelly_fraction=0.25):
    """
    Entraîne un modèle sur le passé et parie sur le futur (les donnees de test)
    """
    data, le_team, le_res = preparer_donnees_ml(df_original)

    # On utilise les codes équipes et les cotes comme indicateurs
    features = ['HomeTeam_Code', 'AwayTeam_Code', 'B365H', 'B365D', 'B365A']
    X = data[features]
    y = data['FTR_Code'] # Ce qu'on veut prédire (0, 1, 2)
    

    # Pas de shuffle=True, on coupe temporellement 
    # On prend les 80% premiers matchs pour apprendre, les 20% derniers pour parier
    split_index = int(len(data) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # On garde les infos de cotes et résultats pour la simulation
    test_data = data.iloc[split_index:].copy()
    
    print(f"Entraînement sur {len(X_train)} matchs. Simulation sur {len(X_test)} matchs.")
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    
    # pred
    predictions_code = model.predict(X_test)
    #calcule le pourcentage d'arbres de la foret qui ont voté pour ce res
    predictions_proba = model.predict_proba(X_test)
    
    # Simulation de Paris
    capital = capital_initial
    historique_capital = [capital]
    nb_paris = 0
    nb_victoires = 0
    montant_total_mise = 0
    
    test_data = test_data.reset_index(drop=True)
    
    # Mapping inverse pour savoir sur quelle cote parier
    # le_res.classes_ donne l'ordre, ex: ['A', 'D', 'H']
    # Il faut mapper le code prédit (0, 1, 2) vers la colonne de cote (B365A, B365D, B365H)
    
    for i, row in test_data.iterrows():
            
        pred_code = predictions_code[i]
        # On récup la proba de la classe qui a été choisie par le modèle
        confiance_du_modele = max(predictions_proba[i])
        
        if confiance_du_modele < seuil_confiance:
            continue

        pred_label = le_res.inverse_transform([pred_code])[0]
        
        # Recup de la cote associée à la prédiction
        cote_du_pari = 0
        if pred_label == 'H':
            cote_du_pari = row['B365H']
        elif pred_label == 'D':
            cote_du_pari = row['B365D']
        elif pred_label == 'A':
            cote_du_pari = row['B365A']
            
        # Si cote invalide, on passe
        if pd.isna(cote_du_pari):
            continue
        
        if use_kelly:
            b = cote_du_pari - 1.0
            p = confiance_du_modele
            q = 1.0 - p
            f_star = p - (q / b)
            
            if f_star > 0: # S'il y a une "Value" mathématique
                mise = capital * f_star * kelly_fraction
                mise = min(mise, capital * 0.15) # Plafond de sécu à 15%
                montant_total_mise+=mise
            else:
                mise = mise # L'IA n'a pas trouvé d'avantage par rapport à la cote, on passe.
                montant_total_mise+=mise
                
        if capital < mise:
            print("Faillite atteinte.")
            break
            
        # Placement du pari
        capital -= mise
        nb_paris += 1
        
        if row['FTR'] == pred_label:
            capital += mise * cote_du_pari
            nb_victoires += 1
            
        historique_capital.append(capital)
    
    if use_kelly:
        return calcul_resultats(capital_initial, capital, nb_paris, nb_victoires, historique_capital, mise, use_kelly, montant_total_mise)
    return calcul_resultats(capital_initial, capital, nb_paris, nb_victoires, historique_capital, mise)

if __name__ == "__main__":
    df = pd.read_csv('../charger_data/epl_historique.csv')

    res = simulation_ml_strategy(df, capital_initial=100, mise=1, seuil_confiance = 0.0)

    print(f"\n--- Res ---")
    print(f"Capital Final : {res['capital_final']} €")
    print(f"Profit        : {res['profit']} €")
    print(f"ROI           : {res['roi']} %")
    print(f"Win Rate      : {res['win_rate']} %")
    print(f"Nb Paris      : {res['nb_paris']}")

    plt.figure(figsize=(10, 6))
    plt.plot(res['historique'], label='Capital', color='purple')
    plt.axhline(y=100, color='r', linestyle='--', label='Départ')
    plt.title('Random Forest')
    plt.xlabel('Nombre de paris')
    plt.ylabel('Capital')
    plt.legend()
    plt.grid(True)
    plt.show()
