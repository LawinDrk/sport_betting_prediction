import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Importations des stratégies heuristiques et du manager de données
from heuristic.heuristic_homeAwayDraw import *
from heuristic.heuristic_risk import *
from heuristic.heuristic_custom import *
from machine_learning.random_forest_away_strategy import simulation_ai_with_strategy

class OngletSimulation:
    """
    Classe gérant l'onglet 'Simulation Stratégies'.
    Permet de configurer un capital initial et une mise fixe, de sélectionner une stratégie
    heuristique ou d'IA, de lancer la simulation sur l'historique de Premier League et
    de tracer la courbe d'évolution du capital.
    """
    def __init__(self, notebook, df, colors):
        self.notebook = notebook
        self.df = df
        self.COLORS = colors
        
        # Création et ajout de l'onglet
        self.frame = tk.Frame(self.notebook)
        self.notebook.add(self.frame, text="Simulation Stratégies")
        
        self.creer_interface()
        
    def creer_interface(self):
        """
        Crée les éléments visuels de l'onglet (panneau de contrôle à gauche, graphique à droite).
        """
        # Panneau latéral gauche pour la configuration
        panneau_gauche = tk.Frame(self.frame, padx=20, pady=20, width=300, bg=self.COLORS["bg_dark"])
        panneau_gauche.pack(side=tk.LEFT, fill=tk.Y)
        
        lbl_style = {"bg": self.COLORS["bg_dark"], "fg": self.COLORS["fg_text"]}
        entry_style = {"bg": self.COLORS["bg_card"], "fg": "white", "insertbackground": "white", "relief": "flat"}
        
        tk.Label(panneau_gauche, text="Configuration", font=("Segoe UI", 16, "bold"), **lbl_style).pack(pady=(0, 20))
        
        # Paramètres Capital & Mise
        tk.Label(panneau_gauche, text="Capital Initial (€):", **lbl_style).pack(anchor="w")
        self.entree_capital = tk.Entry(panneau_gauche, **entry_style)
        self.entree_capital.insert(0, "100")
        self.entree_capital.pack(fill=tk.X, pady=(0, 10), ipady=3)
        
        tk.Label(panneau_gauche, text="Mise par pari (€):", **lbl_style).pack(anchor="w")
        self.entree_mise = tk.Entry(panneau_gauche, **entry_style)
        self.entree_mise.insert(0, "1")
        self.entree_mise.pack(fill=tk.X, pady=(0, 20), ipady=3)
        
        tk.Label(panneau_gauche, text="Choisir une Stratégie :", font=("Segoe UI", 12, "bold"), **lbl_style).pack(anchor="w", pady=(10, 5))
        
        # Liste des stratégies disponibles
        self.choix_strategie = tk.StringVar(value="home")
        strategies = [
            ("Miser Domicile (Home)", "home"),
            ("Miser Extérieur (Away)", "away"),
            ("Miser Match Nul (Draw)", "draw"),
            ("Miser Favori (Cote Min)", "low_risk"),
            ("Miser Outsider (Cote Max)", "high_risk"),
            ("Stratégie Optimisée (Away 3.0-4.0)", "custom"),
            ("IA + Stratégie Away (3.0-4.0)", "ai_strategy")
        ]
        
        for text, val in strategies:
            tk.Radiobutton(panneau_gauche, text=text, variable=self.choix_strategie, value=val, anchor="w",
                           bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_sub"], selectcolor=self.COLORS["bg_card"], 
                           activebackground=self.COLORS["bg_dark"]).pack(fill=tk.X, pady=2)
            
        # Bouton de lancement
        tk.Button(panneau_gauche, text="LANCER SIMULATION", bg=self.COLORS["accent"], fg="white", 
                  font=("Segoe UI", 12, "bold"), relief="flat", activebackground=self.COLORS["accent_hover"], 
                  activeforeground="white", command=self.lancer_simulation).pack(fill=tk.X, pady=30, ipady=5)
        
        # Cadre d'affichage des résultats clés
        labelframe_res = tk.LabelFrame(panneau_gauche, text="Résultats Clés", font=("Segoe UI", 10, "bold"), 
                                       padx=10, pady=10, bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_text"])
        labelframe_res.pack(fill=tk.X)
        
        self.lbl_res_capital = tk.Label(labelframe_res, text="Capital Final : - €", fg=self.COLORS["accent"], 
                                         bg=self.COLORS["bg_dark"], font=("Segoe UI", 11))
        self.lbl_res_capital.pack(anchor="w")
        self.lbl_res_roi = tk.Label(labelframe_res, text="ROI : - %", font=("Segoe UI", 11), 
                                     bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_text"])
        self.lbl_res_roi.pack(anchor="w")
        self.lbl_res_win = tk.Label(labelframe_res, text="Taux Victoire : - %", font=("Segoe UI", 11), 
                                     bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_text"])
        self.lbl_res_win.pack(anchor="w")
        self.lbl_res_nb = tk.Label(labelframe_res, text="Nombre Paris : -", font=("Segoe UI", 10), 
                                    bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_sub"])
        self.lbl_res_nb.pack(anchor="w", pady=(5,0))
        
        self.lbl_sim_status = tk.Label(panneau_gauche, text="", font=("Segoe UI", 10, "italic"), 
                                        bg=self.COLORS["bg_dark"], fg=self.COLORS["warning"])
        self.lbl_sim_status.pack(fill=tk.X, pady=(10, 0))
        
        # Panneau droit pour afficher le graphique Matplotlib
        panneau_droit = tk.Frame(self.frame, bg="white")
        panneau_droit.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("En attente de simulation...")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=panneau_droit)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def lancer_simulation(self):
        """
        Vérifie les entrées utilisateur et lance la simulation sélectionnée dans un thread séparé.
        """
        try:
            capital = float(self.entree_capital.get())
            mise = float(self.entree_mise.get())
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des nombres valides pour le capital et la mise.")
            return

        if self.df is None:
            messagebox.showerror("Erreur", "Données non chargées.")
            return

        strat = self.choix_strategie.get()
        
        def run_sim():
            self.frame.after(0, lambda: self.lbl_sim_status.config(text="Simulation en cours..."))
            try:
                resultats = {}
                if strat == "home": 
                    resultats = simulation_classique(self.df, capital, mise, winner='H')
                elif strat == "away": 
                    resultats = simulation_classique(self.df, capital, mise, winner='A')
                elif strat == "draw": 
                    resultats = simulation_classique(self.df, capital, mise, winner='D')
                elif strat == "low_risk": 
                    resultats = simulation_risque(self.df, capital, mise, strategy='lowest')
                elif strat == "high_risk": 
                    resultats = simulation_risque(self.df, capital, mise, strategy='highest')
                elif strat == "custom": 
                    resultats = simulation_custom(self.df, capital, mise)
                elif strat == "ai_strategy": 
                    resultats = simulation_ai_with_strategy(self.df, capital, mise)
                
                self.frame.after(0, lambda: self.afficher_resultats_simulation(resultats, capital, strat))
                self.frame.after(0, lambda: self.lbl_sim_status.config(text="Terminé ✅", fg=self.COLORS["success"]))
                
            except Exception as e:
                self.frame.after(0, lambda: self.lbl_sim_status.config(text="Erreur ❌", fg=self.COLORS["danger"]))
                self.frame.after(0, lambda: messagebox.showerror("Erreur Simulation", f"Erreur:\n{e}"))

        threading.Thread(target=run_sim, daemon=True).start()

    def afficher_resultats_simulation(self, resultats, capital_initial, strat):
        """
        Met à jour les étiquettes de résultats et dessine le graphique Matplotlib avec les nouvelles données.
        """
        self.lbl_res_capital.config(text=f"Capital Final : {resultats['capital_final']} €")
        self.lbl_res_roi.config(text=f"ROI : {resultats['roi']} %", fg="green" if resultats['roi'] >= 0 else "red")
        self.lbl_res_win.config(text=f"Taux Victoire : {resultats['win_rate']} %")
        self.lbl_res_nb.config(text=f"Nombre Paris : {resultats['nb_paris']}")
        
        self.ax.clear()
        self.ax.plot(resultats['historique'], label='Capital', color='#1f77b4')
        self.ax.axhline(y=capital_initial, color='r', linestyle='--', label='Capital Initial')
        self.ax.set_title(f"Évolution du Capital")
        self.ax.set_xlabel("Nombre de paris")
        self.ax.set_ylabel("Capital (€)")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()
