import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from fonctions_utilitaires import preparer_donnees_ml

def generer_matrice_confusion(df_original):
    data, le_team, le_res = preparer_donnees_ml(df_original)
    
    features = ['HomeTeam_Code', 'AwayTeam_Code', 'B365H', 'B365D', 'B365A']
    X = data[features]
    y = data['FTR_Code']
    
    split_index = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    matrice = confusion_matrix(y_test, y_pred)
    
    labels_utilises = le_res.inverse_transform(sorted(data['FTR_Code'].unique()))

    print(classification_report(y_test, y_pred, target_names=labels_utilises))
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        matrice, 
        annot=True,       
        fmt="d",             
        cmap="Blues",        
        xticklabels=labels_utilises, 
        yticklabels=labels_utilises
    )
    
    plt.title("Matrice de Confusion - RF avec n_est=500, depth=6, seed=42")
    plt.ylabel("Vraie Réalité")
    plt.xlabel("Prédiction du RF ")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('../charger_data/epl_historique.csv')
    generer_matrice_confusion(df)