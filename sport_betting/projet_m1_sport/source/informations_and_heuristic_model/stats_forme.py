import pandas as pd
def calculer_forme_dernier_matchs(df, nb_matchs=5):
    """Calcule les stats des équipes sur leurs N derniers matchs (incluant tirs et fautes)"""
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by='Date')
    
    equipes = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    stats_list = []
    
    for equipe in equipes:
        matchs_equipe = df[(df['HomeTeam'] == equipe) | (df['AwayTeam'] == equipe)].tail(nb_matchs)
        
        v = n = d = bp = bc = 0
        tirs = tirs_cadres = fautes = 0
        
        for _, row in matchs_equipe.iterrows():
            if row['HomeTeam'] == equipe:
                bp += row['FTHG']
                bc += row['FTAG']
                tirs += row['HS']
                tirs_cadres += row['HST']
                fautes += row['HF']
                if row['FTR'] == 'H': v += 1
                elif row['FTR'] == 'D': n += 1
                else: d += 1
            else:
                bp += row['FTAG']
                bc += row['FTHG']
                tirs += row['AS']
                tirs_cadres += row['AST']
                fautes += row['AF']
                if row['FTR'] == 'A': v += 1
                elif row['FTR'] == 'D': n += 1
                else: d += 1
                
        pts = (v * 3) + (n * 1)
        diff = bp - bc
        
        stats_list.append({
            'Equipe': equipe, 'Pts': pts, 'V': v, 'N': n, 'D': d,
            'BP': bp, 'BC': bc, 'Diff': diff,
            'Tirs': tirs, 'TC': tirs_cadres, 'Fautes': fautes
        })
        
    df_stats = pd.DataFrame(stats_list)
    df_stats = df_stats.sort_values(by=['Pts', 'Diff'], ascending=[False, False])
    return df_stats