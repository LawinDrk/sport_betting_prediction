import pandas as pd
from sklearn.preprocessing import LabelEncoder

def calcul_resultats(cap_init, cap_final, nb_paris, nb_victoires, historique, mise=1, use_kelly=False, montant_total_mise=0):
    """
    Paramètres:
    - cap_init: Capital de départ (ex. 100€)
    - cap_final: Capital final à la fin de la simulation
    - nb_paris: Nombre total de paris placés
    - nb_victoires: Nombre de paris gagnés
    - historique: Liste de l'évolution du capital après chaque pari placé
    - mise: Montant misé par pari (par défaut 1€)
    
    Retourne:
    Un dictionnaire contenant le capital final, le profit net, le ROI (%), 
    le nombre de paris, le taux de victoire (%) et l'historique du capital.
    """
    profit = cap_final - cap_init
    if use_kelly:
        roi = (profit / montant_total_mise) * 100
    else:
        roi = (profit / (nb_paris * mise)) * 100
    win_rate = (nb_victoires / nb_paris) * 100
    
    return {
        "capital_final": round(cap_final, 2),
        "profit": round(profit, 2),
        "roi": round(roi, 2),
        "nb_paris": nb_paris,
        "win_rate": round(win_rate, 2),
        "historique": historique
    }

def preparer_donnees_ml(df):
    """
    Nettoie et transforme les données pour le machine learning.
     Paramètres:
    - df: DataFrame brut contenant les données historiques de Premier League (epl_historique.csv)
    
    Retourne:
    - data: DataFrame nettoyé et encodé
    - le_team: Encodeur de labels (LabelEncoder) pour les noms d'équipes
    - le_res: Encodeur de labels (LabelEncoder) pour le résultat du match (FTR)
    """
    # tri chronologique
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by='Date')
    
    # On garde les équipes, les cotes et le résultat
    cols_to_keep = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR']
    data = df[cols_to_keep].copy()
    
    # nettoyage des lignes avec des vals manquantes
    data = data.dropna()
    
    # encodage des valeurs textuelles HomeTeam, AwayTeam, FTR
    # car l'algo de ML ne comprend que les chiffres.
    le_team = LabelEncoder()
    # On fit sur l'ensemble des équipes Home + Away pour avoir le même ID
    all_teams = pd.concat([data['HomeTeam'], data['AwayTeam']]).unique()
    le_team.fit(all_teams)
    
    data['HomeTeam_Code'] = le_team.transform(data['HomeTeam'])
    data['AwayTeam_Code'] = le_team.transform(data['AwayTeam'])
    
    # encodage de la cible FTR : H=0, D=1,A=2 un truc comme ça 
    le_res = LabelEncoder()
    data['FTR_Code'] = le_res.fit_transform(data['FTR'])
    
    return data, le_team, le_res