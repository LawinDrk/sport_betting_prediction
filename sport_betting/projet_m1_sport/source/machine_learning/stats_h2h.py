import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def predire_confrontation_rf(df, home_team, away_team, odd_h, odd_d, odd_a):
    """Entraîne un RF sur tout l'historique et prédit la proba pour une affiche"""
    df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR']).copy()
    
    le_team = LabelEncoder()
    le_team.fit(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    
    df['HomeTeam_Code'] = le_team.transform(df['HomeTeam'])
    df['AwayTeam_Code'] = le_team.transform(df['AwayTeam'])
    
    le_res = LabelEncoder()
    df['FTR_Code'] = le_res.fit_transform(df['FTR'])
    
    features = ['HomeTeam_Code', 'AwayTeam_Code', 'B365H', 'B365D', 'B365A']
    X = df[features]
    y = df['FTR_Code']
    
    model = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42)
    model.fit(X, y)
    
    try:
        home_code = le_team.transform([home_team])[0]
        away_code = le_team.transform([away_team])[0]
    except ValueError:
        return None, "Une des équipes est inconnue dans l'historique."
        
    X_new = pd.DataFrame([[home_code, away_code, odd_h, odd_d, odd_a]], columns=features)
    
    probs = model.predict_proba(X_new)[0]
    classes = le_res.inverse_transform(model.classes_)
    
    return {c: round(p * 100, 2) for c, p in zip(classes, probs)}, None