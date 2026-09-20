import pandas as pd

def charger_donnees(chemin_csv):
    df = pd.read_csv(chemin_csv)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by='Date')
    
    return df