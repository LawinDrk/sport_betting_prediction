import pandas as pd
from .calc_res import * 

def simulation_custom(df, capital, mise):
    capital_courant = capital
    historique = [capital]
    nb_paris = 0
    nb_victoires = 0
    
    for _, row in df.iterrows():
        if capital_courant < mise:
            break

        # Récupération des cotes Away
        a = row['B365A'] if pd.notna(row.get('B365A')) else row.get('AvgA')
        
        if pd.isna(a):
            continue
            
        # Stratégie: Bet Away si 3.0 <= cote < 4.0
        if 3.0 <= a < 4.0:
            capital_courant -= mise
            nb_paris += 1
        
            if row['FTR'] == 'A':
                capital_courant += mise * a
                nb_victoires += 1
            
            historique.append(capital_courant)

    return calcul_resultats(capital, capital_courant, nb_paris, nb_victoires, historique, mise)
