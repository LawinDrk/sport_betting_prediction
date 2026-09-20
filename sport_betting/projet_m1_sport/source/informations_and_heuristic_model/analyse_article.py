import requests
import pandas as pd
from datetime import datetime, timedelta

from informations_and_heuristic_model.articlev2_scrapper import get_sentiment

# --- CONFIGURATION ---
ODDS_API_KEY = "c7d2418cf727dbc54023e78c9c995cd1"
LEAGUE = "soccer_epl"

def run_prediction_pipeline(days_ahead):
    # 1. Préparation des dates
    target_dt = datetime.now() + timedelta(days=days_ahead)
    target_str = target_dt.strftime('%Y-%m-%d')
    filename = f"{target_str}_odds.csv"

    print(f"--- Pipeline lancée pour le {target_str} ---")

    # 2. Récupération des cotes
    url_odds = f"https://api.the-odds-api.com/v4/sports/{LEAGUE}/odds"
    params = {'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
    
    try:
        res = requests.get(url_odds, params=params).json()
    except Exception as e:
        print(f"Erreur API Odds: {e}")
        return

    results = []
    for event in res:
        if target_str in event['commence_time']:
            home = event['home_team']
            away = event['away_team']
            
            # Extraction des cotes 1N2
            odds_h, odds_d, odds_a = None, None, None
            if event['bookmakers']:
                outcomes = event['bookmakers'][0]['markets'][0]['outcomes']
                for o in outcomes:
                    if o['name'] == home: odds_h = o['price']
                    elif o['name'] == away: odds_a = o['price']
                    else: odds_d = o['price']

            # 3. Calcul Sentiment + Heuristique
            sent_home = get_sentiment(home)
            sent_away = get_sentiment(away)

            # Application de ton heuristique > 0.5
            conseil = "RAS"
            if sent_home > 0.20: conseil = f"BET {home}"
            elif sent_away > 0.20: conseil = f"BET {away}"

            results.append({
                'Date_Match': event['commence_time'],
                'HomeTeam': home,
                'AwayTeam': away,
                'Cote_H': odds_h,
                'Cote_N': odds_d,
                'Cote_A': odds_a,
                'Sent_Home': round(sent_home, 2),
                'Sent_Away': round(sent_away, 2),
                'Decision': conseil
            })

    # 4. Enregistrement CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"✓ Succès ! Fichier généré : {filename}")
        print(df[['HomeTeam', 'AwayTeam', 'Decision']].to_string())
        return len(results)
    else:
        print(f"Aucun match de Premier League trouvé pour le {target_str}.")
        return 0

if __name__ == "__main__":
    run_prediction_pipeline()