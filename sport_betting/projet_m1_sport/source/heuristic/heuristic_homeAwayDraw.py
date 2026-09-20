import pandas as pd
from .calc_res import * 

def simulation_classique(df, capital, mise, winner):
    """
    Stratégie: Home ('H'), Away ('A'), Draw ('D')
    """
    capital_courant = capital
    historique = [capital]
    nb_paris = 0
    nb_victoires = 0
    
    for _, row in df.iterrows():
        if capital_courant < mise:
            break
            
        # Sélection de la cote (B365 ou Moyenne)
        col_b365 = 'B365' + winner
        col_avg = 'Avg' + winner
        
        cote = row[col_b365] if pd.notna(row.get(col_b365)) else row.get(col_avg)
        
        if pd.isna(cote):
            continue
            
        capital_courant -= mise
        nb_paris += 1
        
        if row['FTR'] == winner:
            capital_courant += mise * cote
            nb_victoires += 1
            
        historique.append(capital_courant)
        
    return calcul_resultats(capital, capital_courant, nb_paris, nb_victoires, historique, mise)
