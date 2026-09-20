import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import threading
from informations_and_heuristic_model.stats_forme import calculer_forme_dernier_matchs
from machine_learning.stats_h2h import predire_confrontation_rf

class OngletH2H:
    """
    Classe gérant l'onglet 'Forme & H2H'.
    Affiche l'état de forme des équipes (Tirs, Fautes, etc.) et permet
    de simuler une confrontation spécifique via le Random Forest.
    """
    def __init__(self, notebook, df, colors):
        self.notebook = notebook
        self.df = df
        self.COLORS = colors
        
        self.frame = tk.Frame(self.notebook, bg=self.COLORS["bg_dark"])
        self.notebook.add(self.frame, text="Forme & Prédiction IA")
        
        self.creer_interface()
        
    def creer_interface(self):
        # PARTIE HAUTE : Classement de la Forme 
        frame_top = tk.Frame(self.frame, bg=self.COLORS["bg_dark"])
        frame_top.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(frame_top, text=" État de Forme (5 derniers matchs)", font=("Segoe UI", 14, "bold"), 
                 bg=self.COLORS["bg_dark"], fg=self.COLORS["warning"]).pack(anchor="w", pady=(0, 10))
        
        # Nouvelles colonnes intégrées
        cols = ("Équipe", "Pts", "V", "N", "D", "Buts+", "Buts-", "Diff", "Tirs", "Tirs Cadrés", "Fautes")
        self.tree_forme = ttk.Treeview(frame_top, columns=cols, show='headings', height=8)
        
        for col in cols:
            self.tree_forme.heading(col, text=col)
            largeur = 140 if col == "Équipe" else (80 if col in ["Tirs", "Tirs Cadrés", "Fautes"] else 50)
            self.tree_forme.column(col, anchor="center", width=largeur)
            
        self.tree_forme.pack(fill=tk.BOTH, expand=True)
        
        # PARTIE BASSE : Simulateur Face à Face
        frame_bot = tk.LabelFrame(self.frame, text=" Prédiction IA (Random Forest)", font=("Segoe UI", 12, "bold"), 
                                  bg=self.COLORS["bg_dark"], fg=self.COLORS["accent"], padx=20, pady=20)
        frame_bot.pack(fill=tk.X, padx=20, pady=20)
        
        controls = tk.Frame(frame_bot, bg=self.COLORS["bg_dark"])
        controls.pack(fill=tk.X)
        
        lbl_style = {"bg": self.COLORS["bg_dark"], "fg": self.COLORS["fg_text"]}
        entry_style = {"bg": self.COLORS["bg_card"], "fg": "white", "insertbackground": "white", "relief": "flat", "width": 8}
        
        # Sélection équipes
        tk.Label(controls, text="Domicile :", **lbl_style).grid(row=0, column=0, padx=5, pady=5)
        self.cb_home = ttk.Combobox(controls, state="readonly", width=15)
        self.cb_home.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(controls, text="Extérieur :", **lbl_style).grid(row=0, column=2, padx=5, pady=5)
        self.cb_away = ttk.Combobox(controls, state="readonly", width=15)
        self.cb_away.grid(row=0, column=3, padx=5, pady=5)
        
        # Cotes du bookmaker
        tk.Label(controls, text="Cote Domicile :", **lbl_style).grid(row=1, column=0, padx=5, pady=5)
        self.ent_cote_h = tk.Entry(controls, **entry_style)
        self.ent_cote_h.insert(0, "2.50")
        self.ent_cote_h.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(controls, text="Cote Nul :", **lbl_style).grid(row=1, column=2, padx=5, pady=5)
        self.ent_cote_d = tk.Entry(controls, **entry_style)
        self.ent_cote_d.insert(0, "3.10")
        self.ent_cote_d.grid(row=1, column=3, padx=5, pady=5)
        
        tk.Label(controls, text="Cote Extérieur :", **lbl_style).grid(row=1, column=4, padx=5, pady=5)
        self.ent_cote_a = tk.Entry(controls, **entry_style)
        self.ent_cote_a.insert(0, "2.80")
        self.ent_cote_a.grid(row=1, column=5, padx=5, pady=5)
        
        # Bouton
        self.btn_predire = tk.Button(controls, text="PRÉDIRE LE MATCH", bg=self.COLORS["accent"], fg="white", 
                                     font=("Segoe UI", 10, "bold"), command=self.lancer_prediction_h2h)
        self.btn_predire.grid(row=0, column=6, rowspan=2, padx=20)
        
        # Affichage des pourcentages
        self.lbl_proba_h = tk.Label(frame_bot, text="Victoire Domicile : --%", font=("Segoe UI", 14), bg=self.COLORS["bg_dark"], fg=self.COLORS["success"])
        self.lbl_proba_h.pack(side=tk.LEFT, expand=True, pady=15)
        
        self.lbl_proba_d = tk.Label(frame_bot, text="Match Nul : --%", font=("Segoe UI", 14), bg=self.COLORS["bg_dark"], fg=self.COLORS["warning"])
        self.lbl_proba_d.pack(side=tk.LEFT, expand=True, pady=15)
        
        self.lbl_proba_a = tk.Label(frame_bot, text="Victoire Extérieur : --%", font=("Segoe UI", 14), bg=self.COLORS["bg_dark"], fg=self.COLORS["danger"])
        self.lbl_proba_a.pack(side=tk.LEFT, expand=True, pady=15)

        # Remplissage automatique des données une fois l'interface chargée
        self.frame.after(500, self.initialiser_donnees_h2h)

    def initialiser_donnees_h2h(self):
        """Remplit le tableau des formes et les listes déroulantes"""
        if self.df is not None and not self.df.empty:
            df_forme = calculer_forme_dernier_matchs(self.df, nb_matchs=5)
            
            for item in self.tree_forme.get_children(): 
                self.tree_forme.delete(item)
                
            for _, row in df_forme.iterrows():
                self.tree_forme.insert("", "end", values=(
                    row['Equipe'], row['Pts'], row['V'], row['N'], row['D'], 
                    row['BP'], row['BC'], row['Diff'], 
                    row['Tirs'], row['TC'], row['Fautes']
                ))
                
            equipes = sorted(list(pd.concat([self.df['HomeTeam'], self.df['AwayTeam']]).unique()))
            self.cb_home['values'] = equipes
            self.cb_away['values'] = equipes
            if len(equipes) > 1:
                self.cb_home.current(0)
                self.cb_away.current(1)

    def lancer_prediction_h2h(self):
        try:
            h_team = self.cb_home.get()
            a_team = self.cb_away.get()
            odd_h = float(self.ent_cote_h.get())
            odd_d = float(self.ent_cote_d.get())
            odd_a = float(self.ent_cote_a.get())
            
            if h_team == a_team:
                messagebox.showwarning("Attention", "Veuillez choisir deux équipes différentes.")
                return
            
            self.btn_predire.config(state=tk.DISABLED, text="CALCUL EN COURS...")
            
            # Utilisation d'un Thread pour ne pas figer l'interface pendant l'entraînement du modèle
            def run_pred():
                probs, erreur = predire_confrontation_rf(self.df, h_team, a_team, odd_h, odd_d, odd_a)
                
                if erreur:
                    self.frame.after(0, lambda: messagebox.showerror("Erreur", erreur))
                else:
                    self.frame.after(0, lambda: self.lbl_proba_h.config(text=f"Victoire {h_team} : {probs.get('H', 0)}%"))
                    self.frame.after(0, lambda: self.lbl_proba_d.config(text=f"Match Nul : {probs.get('D', 0)}%"))
                    self.frame.after(0, lambda: self.lbl_proba_a.config(text=f"Victoire {a_team} : {probs.get('A', 0)}%"))
                    
                self.frame.after(0, lambda: self.btn_predire.config(state=tk.NORMAL, text="PRÉDIRE LE MATCH"))

            threading.Thread(target=run_pred, daemon=True).start()
                
        except ValueError:
            messagebox.showerror("Erreur", "Les cotes doivent être des nombres valides (ex: 2.50).")