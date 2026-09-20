import pandas as pd
from .calc_res import * 

def simulation_risque(df, capital, mise, strategy):
    """
    Stratégie: Lowest (Favori) ou Highest (Outsider)
    """
    capital_courant = capital
    historique = [capital]
    nb_paris = 0
    nb_victoires = 0
    
    for _, row in df.iterrows():
        if capital_courant < mise:
            break

        # Récup cotes H et A
        h = row['B365H'] if pd.notna(row.get('B365H')) else row.get('AvgH')
        a = row['B365A'] if pd.notna(row.get('B365A')) else row.get('AvgA')
        
        if pd.isna(h) or pd.isna(a):
            continue
            
        choix = None
        cote_choisie = 0
        
        if strategy == 'lowest':
            # Favori
            if h < a: choix, cote_choisie = 'H', h
            elif a < h: choix, cote_choisie = 'A', a
        elif strategy == 'highest':
            # Outsider
            if h > a: choix, cote_choisie = 'H', h
            elif a > h: choix, cote_choisie = 'A', a
            
        if choix is None:
            continue
            
        capital_courant -= mise
        nb_paris += 1
        
        if row['FTR'] == choix:
            capital_courant += mise * cote_choisie
            nb_victoires += 1
            
        historique.append(capital_courant)

    return calcul_resultats(capital, capital_courant, nb_paris, nb_victoires, historique, mise)

