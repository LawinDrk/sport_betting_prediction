import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "epl_historique.csv")

def ajouter_forme_dataframe(df):
    """
    Calcule la forme 
    sur un DataFrame Pandas déjà chargé en mémoire.
    """
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True, errors='coerce')
    df_copy = df_copy.sort_values(by='Date').dropna(subset=['FTR', 'HomeTeam', 'AwayTeam'])
    
    historique_equipes = {team: [] for team in pd.concat([df_copy['HomeTeam'], df_copy['AwayTeam']]).unique()}
    
    forme_home = []
    forme_away = []
    
    for idx, row in df_copy.iterrows():
        h_team = row['HomeTeam']
        a_team = row['AwayTeam']
        
        forme_home.append(sum(historique_equipes[h_team][-5:]))
        forme_away.append(sum(historique_equipes[a_team][-5:]))
        
        if row['FTR'] == 'H':
            historique_equipes[h_team].append(3)
            historique_equipes[a_team].append(0)
        elif row['FTR'] == 'A':
            historique_equipes[h_team].append(0)
            historique_equipes[a_team].append(3)
        else:
            historique_equipes[h_team].append(1)
            historique_equipes[a_team].append(1)
            
    df_copy['Forme_Home'] = forme_home
    df_copy['Forme_Away'] = forme_away
    df_copy['Forme_Diff'] = df_copy['Forme_Home'] - df_copy['Forme_Away']
    
    return df_copy

def calculer_forme_historique(chemin_csv):
    """
    Charge le CSV depuis le disque et applique le calcul.
    """
    if not os.path.exists(chemin_csv):
        print("Fichier historique introuvable.")
        return None
        
    print("Calcul de la forme historique équipe par équipe...")
    df = pd.read_csv(chemin_csv)
    return ajouter_forme_dataframe(df)

if __name__ == "__main__":
    df_enrichi = calculer_forme_historique(CSV_FILE)
    if df_enrichi is not None:
        df_enrichi.to_csv("epl_historique_avec_forme.csv", index=False)
        print("Fichier historique enrichi généré.")