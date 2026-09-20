import tkinter as tk
from tkinter import ttk
import requests
import math
import pandas as pd
import threading
import queue
from machine_learning.random_forest_away_strategy import load_or_train_model

API_KEY = "c7d2418cf727dbc54023e78c9c995cd1"
LEAGUE = "soccer_epl"

class OngletCotes:
    """
    Classe gérant l'onglet 'Cotes & Scores Impliqués'.
    Cette classe s'occupe de charger le modèle d'IA pré-entraîné (Random Forest),
    d'initialiser l'interface graphique pour mettre à jour et afficher les cotes live (Winamax / Betclic),
    d'estimer mathématiquement les scores les plus probables via une loi de Poisson,
    et de générer des recommandations de paris basées sur l'espérance mathématique calculée par le modèle.
    
    Toutes les opérations d'API s'exécutent en arrière-plan et communiquent de manière sécurisée
    avec le thread principal via une queue de messages.
    """
    def __init__(self, notebook, data_manager, colors):
        self.notebook = notebook
        self.data_manager = data_manager
        self.COLORS = colors
        
        # Chargement du modèle AI pré-entraîné et des encodeurs
        self.rf_model = None
        self.le_team = None
        self.le_res = None
        self.load_ai_model()
        
        # Conteneur principal de l'onglet
        self.frame = tk.Frame(self.notebook)
        self.notebook.add(self.frame, text="Cotes & Scores Impliqués")
        
        # Initialisation de la queue thread-safe pour la communication avec le thread GUI
        self.queue = queue.Queue()
        
        self.creer_interface()
        self.check_queue()
        
    def load_ai_model(self):
        """
        Charge le modèle Random Forest pré-entraîné et les encodeurs à partir du fichier pickle mis en cache.
        """
        try:
            self.rf_model, self.le_team, self.le_res = load_or_train_model()
            print("[Modèle IA] Chargement réussi du modèle pré-entraîné pour les prédictions en direct.")
        except Exception as e:
            print(f"[Erreur Modèle IA] Impossible de charger le modèle pré-entraîné : {e}")

    def check_queue(self):
        """
        Vérifie régulièrement s'il y a des messages (données de matchs ou erreurs) dans la queue
        envoyés par le thread d'arrière-plan, et met à jour l'interface graphique sur le thread principal.
        """
        try:
            while True:
                msg = self.queue.get_nowait()
                msg_type = msg.get("type")
                if msg_type == "events":
                    self.update_treeview(msg["events"], is_mock=msg.get("is_mock", False))
                elif msg_type == "status":
                    self.lbl_status.config(text=msg["text"], fg=msg["fg"])
        except queue.Empty:
            pass
        self.frame.after(100, self.check_queue)

    def creer_interface(self):
        """
        Crée et positionne les éléments de l'interface utilisateur pour cet onglet (bouton de rafraîchissement, tableau des matchs).
        """
        # Contrôles de l'en-tête
        header_frame = tk.Frame(self.frame, bg=self.COLORS["bg_dark"], pady=15, padx=20)
        header_frame.pack(fill=tk.X)
        
        btn_config = {
            "bg": self.COLORS["accent"], 
            "fg": "white",
            "activebackground": self.COLORS["accent_hover"], 
            "activeforeground": "white",
            "relief": "flat", 
            "font": ("Segoe UI", 10, "bold"),
            "padx": 15,
            "pady": 5
        }
        
        btn_load = tk.Button(header_frame, text="🔄 METTRE À JOUR COTES LIVE (WINAMAX / BETCLIC)", **btn_config, 
                             command=self.charger_cotes_live)
        btn_load.pack(side=tk.LEFT)
        
        self.lbl_status = tk.Label(header_frame, text="Prêt", bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_sub"], font=("Segoe UI", 10))
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        # Tableau des matchs
        columns = ("Date", "Home", "Away", "Winamax_H", "Winamax_D", "Winamax_A", "Betclic_H", "Betclic_D", "Betclic_A", "Score_Implique", "AI_Recommendation")
        self.tree = ttk.Treeview(self.frame, columns=columns, show='headings')
        
        for col in columns:
            title = col.replace("_", " ")
            self.tree.heading(col, text=title)
            if col in ["Date"]:
                self.tree.column(col, width=130, anchor="center")
            elif col in ["Home", "Away"]:
                self.tree.column(col, width=130, anchor="w")
            elif col in ["Score_Implique", "AI_Recommendation"]:
                self.tree.column(col, width=160, anchor="center")
            else:
                self.tree.column(col, width=85, anchor="center")

        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 20))
        
    def normalize_team_name(self, name):
        """
        Normalise le nom d'une équipe pour le faire correspondre à la nomenclature utilisée
        par le modèle d'intelligence artificielle.

        Args:
            name (str): Nom d'origine de l'équipe.

        Returns:
            str: Nom normalisé de l'équipe s'il est présent dans le dictionnaire, sinon le nom d'origine.
        """
        normalization = {
            "Manchester City": "Man City",
            "Manchester United": "Man United",
            "Tottenham Hotspur": "Tottenham",
            "Wolverhampton Wanderers": "Wolves",
            "Nottingham Forest": "Nott'm Forest",
            "Sheffield United": "Sheffield United",
            "Sheffield Utd": "Sheffield United",
            "Leicester City": "Leicester",
            "Leeds United": "Leeds",
            "Luton Town": "Luton",
            "West Ham United": "West Ham",
            "Newcastle United": "Newcastle",
            "Brighton and Hove Albion": "Brighton",
            "Brighton & Hove Albion": "Brighton",
            "Aston Villa": "Aston Villa",
            "Crystal Palace": "Crystal Palace",
            "Arsenal": "Arsenal",
            "Chelsea": "Chelsea",
            "Everton": "Everton",
            "Liverpool": "Liverpool",
            "Fulham": "Fulham",
            "Brentford": "Brentford",
            "Bournemouth": "Bournemouth",
            "Burnley": "Burnley"
        }
        return normalization.get(name, name)

    def estimate_exact_score(self, odds_h, odds_d, odds_a, avg_total_goals=2.7):
        """
        Utilise un modèle de Poisson sur le nombre de buts pour estimer mathématiquement le score
        le plus probable à partir des cotes à domicile, nul et à l'extérieur.

        Args:
            odds_h (float): Cote de la victoire à domicile.
            odds_d (float): Cote du match nul.
            odds_a (float): Cote de la victoire à l'extérieur.
            avg_total_goals (float, optionnel): Nombre moyen attendu de buts dans un match. Par défaut 2.7.

        Returns:
            str: Score estimé sous la forme "ButsDomicile-ButsExtérieur" (ex: "2-1").
        """
        try:
            inv_h = 1.0 / odds_h if odds_h else 0
            inv_d = 1.0 / odds_d if odds_d else 0
            inv_a = 1.0 / odds_a if odds_a else 0
            
            total = inv_h + inv_d + inv_a
            if total == 0:
                return "1-1"
                
            p_h = inv_h / total
            p_d = inv_d / total
            p_a = inv_a / total
            
            # Estimer le nombre attendu de buts en fonction des ratios de cotes
            sum_ha = p_h + p_a if (p_h + p_a) > 0 else 1.0
            lambda_h = avg_total_goals * (p_h / sum_ha)
            lambda_a = avg_total_goals * (p_a / sum_ha)
            
            # Ajuster légèrement les buts selon la probabilité de match nul
            lambda_h *= (1.0 - 0.4 * p_d)
            lambda_a *= (1.0 - 0.4 * p_d)
            
            best_score = "1-1"
            best_prob = -1.0
            
            for h in range(5):
                for a in range(5):
                    prob_h = (math.pow(lambda_h, h) * math.exp(-lambda_h)) / math.factorial(h)
                    prob_a = (math.pow(lambda_a, a) * math.exp(-lambda_a)) / math.factorial(a)
                    prob = prob_h * prob_a
                    
                    # Augmenter/réduire le score de match nul pour s'aligner sur les cotes de match nul
                    if h == a:
                        prob *= (p_d / 0.26)
                        
                    if prob > best_prob:
                        best_prob = prob
                        best_score = f"{h}-{a}"
            return best_score
        except Exception:
            return "1-1"

    def calculate_ai_recommendation(self, home, away, odds_h, odds_d, odds_a):
        """
        Exécute la prédiction en direct sur le modèle Random Forest entraîné pour le match donné
        et vérifie si le pari à l'extérieur est rentable selon l'espérance mathématique (EV).

        Args:
            home (str): Nom de l'équipe à domicile.
            away (str): Nom de l'équipe à l'extérieur.
            odds_h (float): Cote pour la victoire à domicile.
            odds_d (float): Cote pour le match nul.
            odds_a (float): Cote pour la victoire à l'extérieur.

        Returns:
            str: Recommandation de pari ou message de statut de l'opportunité.
        """
        if not self.rf_model or not self.le_team or not self.le_res:
            return "Modèle indisponible"
            
        norm_home = self.normalize_team_name(home)
        norm_away = self.normalize_team_name(away)
        
        try:
            # Encoder les codes des équipes
            if norm_home not in self.le_team.classes_ or norm_away not in self.le_team.classes_:
                return "Équipe non encodée"
                
            home_code = self.le_team.transform([norm_home])[0]
            away_code = self.le_team.transform([norm_away])[0]
            
            # Prédiction
            X_live = pd.DataFrame([{
                'HomeTeam_Code': home_code,
                'AwayTeam_Code': away_code,
                'B365H': odds_h,
                'B365D': odds_d,
                'B365A': odds_a
            }])
            
            probs = self.rf_model.predict_proba(X_live)[0]
            away_class_idx = list(self.le_res.classes_).index('A')
            prob_away = probs[away_class_idx]
            
            # Logique de stratégie de pari : 3.0 <= cotes <= 4.0 et probabilité * cote >= 1.05
            if 3.0 <= odds_a <= 4.0:
                expected_value = prob_away * odds_a
                if expected_value >= 1.05:
                    return f"BET Extérieur ({expected_value:.2f} EV)"
                else:
                    return f"Non rentable ({expected_value:.2f} EV)"
            return "Cote hors limites"
        except Exception as e:
            return "Erreur prediction"

    def obtenir_matchs_simules(self):
        """
        Génère une liste de 20 matchs fictifs de Premier League pour faire fonctionner l'interface
        hors saison (quand l'API renvoie une liste vide).

        Returns:
            list: Liste de 20 dictionnaires représentant les matchs et leurs cotes.
        """
        from datetime import datetime, timedelta
        maintenant = datetime.now()
        
        matchs = [
            # Journée 1
            {
                "home_team": "Manchester City",
                "away_team": "Arsenal",
                "commence_time": (maintenant + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Manchester City", "price": 1.90}, {"name": "Arsenal", "price": 3.60}, {"name": "Draw", "price": 3.70}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Manchester City", "price": 1.88}, {"name": "Arsenal", "price": 3.65}, {"name": "Draw", "price": 3.65}]}]}
                ]
            },
            {
                "home_team": "Chelsea",
                "away_team": "Liverpool",
                "commence_time": (maintenant + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Chelsea", "price": 2.45}, {"name": "Liverpool", "price": 2.70}, {"name": "Draw", "price": 3.40}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Chelsea", "price": 2.40}, {"name": "Liverpool", "price": 2.75}, {"name": "Draw", "price": 3.35}]}]}
                ]
            },
            {
                "home_team": "Aston Villa",
                "away_team": "Manchester United",
                "commence_time": (maintenant + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Aston Villa", "price": 2.15}, {"name": "Manchester United", "price": 3.10}, {"name": "Draw", "price": 3.50}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Aston Villa", "price": 2.10}, {"name": "Manchester United", "price": 3.15}, {"name": "Draw", "price": 3.45}]}]}
                ]
            },
            {
                "home_team": "Everton",
                "away_team": "Tottenham Hotspur",
                "commence_time": (maintenant + timedelta(days=1, hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Everton", "price": 3.20}, {"name": "Tottenham Hotspur", "price": 2.10}, {"name": "Draw", "price": 3.40}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Everton", "price": 3.25}, {"name": "Tottenham Hotspur", "price": 2.05}, {"name": "Draw", "price": 3.45}]}]}
                ]
            },
            {
                "home_team": "Newcastle United",
                "away_team": "Wolverhampton Wanderers",
                "commence_time": (maintenant + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Newcastle United", "price": 1.65}, {"name": "Wolverhampton Wanderers", "price": 4.80}, {"name": "Draw", "price": 4.00}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Newcastle United", "price": 1.62}, {"name": "Wolverhampton Wanderers", "price": 4.90}, {"name": "Draw", "price": 3.95}]}]}
                ]
            },
            {
                "home_team": "West Ham United",
                "away_team": "Crystal Palace",
                "commence_time": (maintenant + timedelta(days=2, hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "West Ham United", "price": 2.00}, {"name": "Crystal Palace", "price": 3.50}, {"name": "Draw", "price": 3.50}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "West Ham United", "price": 1.95}, {"name": "Crystal Palace", "price": 3.55}, {"name": "Draw", "price": 3.45}]}]}
                ]
            },
            {
                "home_team": "Brentford",
                "away_team": "Brighton & Hove Albion",
                "commence_time": (maintenant + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Brentford", "price": 2.30}, {"name": "Brighton & Hove Albion", "price": 2.90}, {"name": "Draw", "price": 3.45}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Brentford", "price": 2.25}, {"name": "Brighton & Hove Albion", "price": 2.95}, {"name": "Draw", "price": 3.40}]}]}
                ]
            },
            {
                "home_team": "Fulham",
                "away_team": "Bournemouth",
                "commence_time": (maintenant + timedelta(days=3, hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Fulham", "price": 2.05}, {"name": "Bournemouth", "price": 3.35}, {"name": "Draw", "price": 3.50}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Fulham", "price": 2.00}, {"name": "Bournemouth", "price": 3.40}, {"name": "Draw", "price": 3.45}]}]}
                ]
            },
            {
                "home_team": "Nottingham Forest",
                "away_team": "Burnley",
                "commence_time": (maintenant + timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Nottingham Forest", "price": 1.95}, {"name": "Burnley", "price": 3.70}, {"name": "Draw", "price": 3.55}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Nottingham Forest", "price": 1.92}, {"name": "Burnley", "price": 3.75}, {"name": "Draw", "price": 3.50}]}]}
                ]
            },
            {
                "home_team": "Leicester City",
                "away_team": "Sheffield United",
                "commence_time": (maintenant + timedelta(days=4, hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Leicester City", "price": 1.80}, {"name": "Sheffield United", "price": 4.20}, {"name": "Draw", "price": 3.65}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Leicester City", "price": 1.78}, {"name": "Sheffield United", "price": 4.30}, {"name": "Draw", "price": 3.60}]}]}
                ]
            },
            # Journée 2
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "commence_time": (maintenant + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Arsenal", "price": 1.75}, {"name": "Chelsea", "price": 4.10}, {"name": "Draw", "price": 3.85}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Arsenal", "price": 1.72}, {"name": "Chelsea", "price": 4.20}, {"name": "Draw", "price": 3.80}]}]}
                ]
            },
            {
                "home_team": "Liverpool",
                "away_team": "Manchester City",
                "commence_time": (maintenant + timedelta(days=7, hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Liverpool", "price": 2.65}, {"name": "Manchester City", "price": 2.45}, {"name": "Draw", "price": 3.60}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Liverpool", "price": 2.70}, {"name": "Manchester City", "price": 2.40}, {"name": "Draw", "price": 3.55}]}]}
                ]
            },
            {
                "home_team": "Manchester United",
                "away_team": "Everton",
                "commence_time": (maintenant + timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Manchester United", "price": 1.55}, {"name": "Everton", "price": 5.40}, {"name": "Draw", "price": 4.10}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Manchester United", "price": 1.52}, {"name": "Everton", "price": 5.60}, {"name": "Draw", "price": 4.00}]}]}
                ]
            },
            {
                "home_team": "Tottenham Hotspur",
                "away_team": "Aston Villa",
                "commence_time": (maintenant + timedelta(days=8, hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Tottenham Hotspur", "price": 1.95}, {"name": "Aston Villa", "price": 3.40}, {"name": "Draw", "price": 3.75}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Tottenham Hotspur", "price": 1.90}, {"name": "Aston Villa", "price": 3.45}, {"name": "Draw", "price": 3.70}]}]}
                ]
            },
            {
                "home_team": "Wolverhampton Wanderers",
                "away_team": "West Ham United",
                "commence_time": (maintenant + timedelta(days=9)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Wolverhampton Wanderers", "price": 2.50}, {"name": "West Ham United", "price": 2.65}, {"name": "Draw", "price": 3.40}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Wolverhampton Wanderers", "price": 2.45}, {"name": "West Ham United", "price": 2.70}, {"name": "Draw", "price": 3.35}]}]}
                ]
            },
            {
                "home_team": "Crystal Palace",
                "away_team": "Newcastle United",
                "commence_time": (maintenant + timedelta(days=9, hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Crystal Palace", "price": 3.10}, {"name": "Newcastle United", "price": 2.15}, {"name": "Draw", "price": 3.50}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Crystal Palace", "price": 3.15}, {"name": "Newcastle United", "price": 2.10}, {"name": "Draw", "price": 3.45}]}]}
                ]
            },
            {
                "home_team": "Brighton & Hove Albion",
                "away_team": "Fulham",
                "commence_time": (maintenant + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Brighton & Hove Albion", "price": 1.85}, {"name": "Fulham", "price": 3.80}, {"name": "Draw", "price": 3.70}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Brighton & Hove Albion", "price": 1.82}, {"name": "Fulham", "price": 3.90}, {"name": "Draw", "price": 3.60}]}]}
                ]
            },
            {
                "home_team": "Bournemouth",
                "away_team": "Brentford",
                "commence_time": (maintenant + timedelta(days=10, hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Bournemouth", "price": 2.05}, {"name": "Brentford", "price": 3.30}, {"name": "Draw", "price": 3.55}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Bournemouth", "price": 2.00}, {"name": "Brentford", "price": 3.35}, {"name": "Draw", "price": 3.50}]}]}
                ]
            },
            {
                "home_team": "Burnley",
                "away_team": "Leicester City",
                "commence_time": (maintenant + timedelta(days=11)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Burnley", "price": 2.90}, {"name": "Leicester City", "price": 2.35}, {"name": "Draw", "price": 3.40}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Burnley", "price": 2.95}, {"name": "Leicester City", "price": 2.30}, {"name": "Draw", "price": 3.35}]}]}
                ]
            },
            {
                "home_team": "Sheffield United",
                "away_team": "Nottingham Forest",
                "commence_time": (maintenant + timedelta(days=11, hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmakers": [
                    {"key": "winamax_fr", "markets": [{"outcomes": [{"name": "Sheffield United", "price": 3.40}, {"name": "Nottingham Forest", "price": 2.05}, {"name": "Draw", "price": 3.45}]}]},
                    {"key": "betclic_fr", "markets": [{"outcomes": [{"name": "Sheffield United", "price": 3.50}, {"name": "Nottingham Forest", "price": 2.00}, {"name": "Draw", "price": 3.40}]}]}
                ]
            }
        ]
        return matchs

    def charger_cotes_live(self):
        """
        Déclenche le chargement asynchrone des cotes live via un appel réseau vers The Odds API.
        Si l'API ne renvoie aucun match (ex: hors saison), bascule sur les matchs simulés de secours.
        """
        self.lbl_status.config(text="Récupération des cotes en cours...", fg=self.COLORS["warning"])
        
        def run():
            url = f"https://api.the-odds-api.com/v4/sports/{LEAGUE}/odds"
            params = {
                'apiKey': API_KEY,
                'regions': 'eu',
                'markets': 'h2h',
                'oddsFormat': 'decimal'
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    events = self.obtenir_matchs_simules()
                    self.queue.put({"type": "events", "events": events, "is_mock": True})
                    return
                    
                events = r.json()
                if not events:
                    events = self.obtenir_matchs_simules()
                    self.queue.put({"type": "events", "events": events, "is_mock": True})
                else:
                    self.queue.put({"type": "events", "events": events, "is_mock": False})
            except Exception as e:
                events = self.obtenir_matchs_simules()
                self.queue.put({"type": "events", "events": events, "is_mock": True})
                
        threading.Thread(target=run, daemon=True).start()

    def update_treeview(self, events, is_mock=False):
        """
        Met à jour le Treeview avec les matchs en direct reçus de l'API (ou simulés), en extrayant les cotes Winamax et Betclic,
        en estimant le score implicite (loi de Poisson) et en affichant les recommandations de l'IA.

        Args:
            events (list): Liste des dictionnaires d'événements contenant les données des matchs et cotes.
            is_mock (bool): Indique si les données proviennent de matchs simulés de secours.
        """
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        count = 0
        for event in events:
            home = event.get('home_team')
            away = event.get('away_team')
            commence_time = event.get('commence_time', '').replace('T', ' ').replace('Z', '')
            
            # Cotes Winamax
            win_h, win_d, win_a = "-", "-", "-"
            # Cotes Betclic
            clic_h, clic_d, clic_a = "-", "-", "-"
            
            for bookmaker in event.get('bookmakers', []):
                key = bookmaker.get('key')
                if key == 'winamax_fr':
                    outcomes = bookmaker.get('markets', [{}])[0].get('outcomes', [])
                    for o in outcomes:
                        if o['name'] == home: win_h = o['price']
                        elif o['name'] == away: win_a = o['price']
                        else: win_d = o['price']
                elif key == 'betclic_fr':
                    outcomes = bookmaker.get('markets', [{}])[0].get('outcomes', [])
                    for o in outcomes:
                        if o['name'] == home: clic_h = o['price']
                        elif o['name'] == away: clic_a = o['price']
                        else: clic_d = o['price']
            
            # Calculer le score implicite de Poisson
            # Utiliser les cotes Winamax par défaut, sinon Betclic
            h_odds = win_h if win_h != "-" else clic_h
            d_odds = win_d if win_d != "-" else clic_d
            a_odds = win_a if win_a != "-" else clic_a
            
            score = "-"
            recommandation = "Cote indisponible"
            
            if h_odds != "-" and d_odds != "-" and a_odds != "-":
                score = self.estimate_exact_score(float(h_odds), float(d_odds), float(a_odds))
                recommandation = self.calculate_ai_recommendation(home, away, float(h_odds), float(d_odds), float(a_odds))
            
            self.tree.insert("", "end", values=(
                commence_time, home, away,
                win_h, win_d, win_a,
                clic_h, clic_d, clic_a,
                score, recommandation
            ))
            count += 1
            
        if is_mock:
            self.lbl_status.config(text=f"Mode Démo : {count} matchs simulés (EPL Hors-Saison / API vide)", fg=self.COLORS["warning"])
        else:
            self.lbl_status.config(text=f"Mise à jour réussie ({count} matchs)", fg=self.COLORS["success"])
