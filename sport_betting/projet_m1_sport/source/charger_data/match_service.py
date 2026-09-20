import pandas as pd
import requests
from io import StringIO
import datetime
import os

class MatchService:
    def __init__(self):
        self.fixtures_url = "https://www.football-data.co.uk/fixtures.csv"

    def get_matches_window(self, weeks_back=2, weeks_forward=2):
        """
        Récupère les matchs dans la fenêtre [maintenant - weeks_back, maintenant + weeks_forward].
        Retourne un DataFrame avec les colonnes : Date, Time, HomeTeam, AwayTeam, B365H, B365D, B365A
        """
        try:
            response = requests.get(self.fixtures_url)
            if response.status_code != 200:
                return pd.DataFrame()

            df = pd.read_csv(StringIO(response.text))
            
            # Fix BOM if present
            df.columns = df.columns.str.replace('ï»¿', '')
            
            # Standardiser les colonnes
            if 'Date' not in df.columns or 'HomeTeam' not in df.columns:
                return pd.DataFrame()


            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date'])

            today = datetime.datetime.now()
            start_date = today - datetime.timedelta(weeks=weeks_back)
            end_date = today + datetime.timedelta(weeks=weeks_forward)

            # Filtrer par fenêtre de date
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            df_filtered = df.loc[mask].copy()

            # Filtrer pour Premier League (Div = E0)
            if 'Div' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['Div'] == 'E0']
            
            # Sélectionner les colonnes pertinentes
            # S'assurer d'avoir l'heure si possible, sinon par défaut 00:00
            if 'Time' not in df_filtered.columns:
                df_filtered['Time'] = ""
            
            cols_to_keep = ['Date', 'Time', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']

            # Ajouter les colonnes de cotes manquantes si elles n'existent pas
            for col in ['B365H', 'B365D', 'B365A']:
                if col not in df_filtered.columns:
                    df_filtered[col] = None

            # Filtrer les colonnes qui existent vraiment
            available_cols = [c for c in cols_to_keep if c in df_filtered.columns]
            df_final = df_filtered[available_cols]

            return df_final.sort_values(by=['Date', 'Time'])

        except Exception as e:
            print(f"[MatchService] Error: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    ms = MatchService()
    df = ms.get_matches_window()
    print(df.head())
    print(df.tail())
