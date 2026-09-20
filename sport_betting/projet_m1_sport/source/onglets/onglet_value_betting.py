from charger_data.charger_donnees import *
from charger_data.data_manager import DataManager


import sys
import os
from informations_and_heuristic_model.analyse_historique_forme import ajouter_forme_dataframe

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class OngletValueBetting:
    """
    Onglet dédié au Value Betting Historique.
    Utilise le Feature Engineering externe (analyse_historique_forme.py)
    pour garantir la non-redondance du code.
    """
    
    def __init__(self, notebook, df, colors):
        self.notebook = notebook
        self.df = df 
        self.COLORS = colors
            
        self.frame = tk.Frame(self.notebook, bg=self.COLORS["bg_dark"])
        self.notebook.add(self.frame, text="Value Betting (Forme)")
            
        self.creer_interface()

    def creer_interface(self):
        # Panneau Gauche : Contrôles
        panneau_gauche = tk.Frame(self.frame, padx=20, pady=20, width=320, bg=self.COLORS["bg_dark"])
        panneau_gauche.pack(side=tk.LEFT, fill=tk.Y)
        
        lbl_style = {"bg": self.COLORS["bg_dark"], "fg": self.COLORS["fg_text"]}
        entry_style = {"bg": self.COLORS["bg_card"], "fg": "white", "insertbackground": "white", "relief": "flat"}
        
        tk.Label(panneau_gauche, text="Value Betting \n(Anomalie de Forme)", font=("Segoe UI", 14, "bold"), **lbl_style).pack(pady=(0, 20))
        
        # Capital
        tk.Label(panneau_gauche, text="Capital Initial (€):", **lbl_style).pack(anchor="w")
        self.entree_capital = tk.Entry(panneau_gauche, **entry_style)
        self.entree_capital.insert(0, "100")
        self.entree_capital.pack(fill=tk.X, pady=(0, 10), ipady=3)
        
        # Mise
        tk.Label(panneau_gauche, text="Mise par pari (€):", **lbl_style).pack(anchor="w")
        self.entree_mise = tk.Entry(panneau_gauche, **entry_style)
        self.entree_mise.insert(0, "1")
        self.entree_mise.pack(fill=tk.X, pady=(0, 15), ipady=3)
        
        # Seuil Différence de Forme
        tk.Label(panneau_gauche, text="Seuil Écart de Forme (pts):", **lbl_style).pack(anchor="w")
        self.entree_seuil = tk.Entry(panneau_gauche, **entry_style)
        self.entree_seuil.insert(0, "3")
        self.entree_seuil.pack(fill=tk.X, pady=(0, 10), ipady=3)
        
        # Cote minimale
        tk.Label(panneau_gauche, text="Cote Minimale acceptée:", **lbl_style).pack(anchor="w")
        self.entree_cote_min = tk.Entry(panneau_gauche, **entry_style)
        self.entree_cote_min.insert(0, "1.80")
        self.entree_cote_min.pack(fill=tk.X, pady=(0, 20), ipady=3)
        
        # Bouton Lancer
        btn_calculer = tk.Button(panneau_gauche, text="ANALYSER L'HISTORIQUE", 
                                 bg=self.COLORS["accent"], fg="white", font=("Segoe UI", 11, "bold"), 
                                 relief="flat", activebackground=self.COLORS["accent_hover"], activeforeground="white",
                                 command=self.executer_backtest)
        btn_calculer.pack(fill=tk.X, pady=(10, 20), ipady=5)
        
        # Zone d'affichage des résultats textuels
        labelframe_res = tk.LabelFrame(panneau_gauche, text="Indicateurs de Performance", font=("Segoe UI", 10, "bold"), padx=10, pady=10,
                                       bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_text"])
        labelframe_res.pack(fill=tk.X)
        
        self.lbl_capital_final = tk.Label(labelframe_res, text="Capital Final : - €", fg=self.COLORS["success"], bg=self.COLORS["bg_dark"], font=("Segoe UI", 11, "bold"))
        self.lbl_capital_final.pack(anchor="w")
        self.lbl_roi = tk.Label(labelframe_res, text="ROI Global : - %", bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_text"], font=("Segoe UI", 11))
        self.lbl_roi.pack(anchor="w")
        self.lbl_winrate = tk.Label(labelframe_res, text="Taux de Réussite : - %", bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_text"], font=("Segoe UI", 11))
        self.lbl_winrate.pack(anchor="w")
        self.lbl_total_bets = tk.Label(labelframe_res, text="Opportunités détectées : -", bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_sub"], font=("Segoe UI", 10))
        self.lbl_total_bets.pack(anchor="w", pady=(5, 0))

        # --- Panneau Droit : Graphique Évolution ---
        self.panneau_droit = tk.Frame(self.frame, bg="white")
        self.panneau_droit.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("En attente de l'analyse rétrospective...")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.panneau_droit)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def executer_backtest(self):
        if self.df is None:
            messagebox.showerror("Erreur", "Données historiques indisponibles.")
            return
            
        try:
            capital = float(self.entree_capital.get())
            mise = float(self.entree_mise.get())
            seuil = int(self.entree_seuil.get())
            cote_min = float(self.entree_cote_min.get())
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez vérifier la validité des paramètres numériques.")
            return

        df_copy = self.df.copy()
        
        # utilisation module externe
        if 'Forme_Diff' not in df_copy.columns:
            try:
                df_copy = ajouter_forme_dataframe(df_copy)
            except Exception as e:
                messagebox.showerror("Erreur Module", f"Impossible d'ajouter la forme via le module externe :\n{e}")
                return

        cap_actuel = capital
        historique_cap = [cap_actuel]
        gagnes, total = 0, 0
        
        # Détection dynamique des colonnes de cotes (selon la source)
        col_h = 'B365H' if 'B365H' in df_copy.columns else ([c for c in df_copy.columns if 'H' in c][0] if [c for c in df_copy.columns if 'H' in c] else None)
        col_a = 'B365A' if 'B365A' in df_copy.columns else ([c for c in df_copy.columns if 'A' in c][0] if [c for c in df_copy.columns if 'A' in c] else None)
        
        if col_h and col_a:
            for idx, row in df_copy.iterrows():
                if pd.isna(row[col_h]) or pd.isna(row[col_a]): continue
                c_h, c_a = float(row[col_h]), float(row[col_a])
                
                # Anomalie Extérieure (Away en forme)
                if (row['Forme_Away'] - row['Forme_Home']) >= seuil and c_a >= cote_min:
                    total += 1
                    if row['FTR'] == 'A':
                        gagnes += 1
                        cap_actuel += mise * (c_a - 1)
                    else:
                        cap_actuel -= mise
                    historique_cap.append(cap_actuel)
                    
                # Anomalie Domicile (Home en forme)
                elif (row['Forme_Home'] - row['Forme_Away']) >= seuil and c_h >= cote_min:
                    total += 1
                    if row['FTR'] == 'H':
                        gagnes += 1
                        cap_actuel += mise * (c_h - 1)
                    else:
                        cap_actuel -= mise
                    historique_cap.append(cap_actuel)

        wr = (gagnes / total * 100) if total > 0 else 0
        roi = ((cap_actuel - capital) / (total * mise) * 100) if total > 0 else 0
        
        self.lbl_capital_final.config(text=f"Capital Final : {cap_actuel:.2f} €")
        self.lbl_roi.config(text=f"ROI Global : {roi:.2f} %")
        self.lbl_winrate.config(text=f"Taux de Réussite : {wr:.2f} %")
        self.lbl_total_bets.config(text=f"Opportunités détectées : {total}")
        
        self.ax.clear()
        self.ax.plot(historique_cap, color=self.COLORS["accent"], linewidth=2, label="Courbe de Capital")
        self.ax.axhline(y=capital, color=self.COLORS["danger"], linestyle="--", label="Capital Initial")
        self.ax.set_title(f"Simulation Value Betting (Seuil: {seuil}, Cote Min: {cote_min})", color=self.COLORS["bg_dark"], fontdict={'weight': 'bold'})
        self.ax.set_xlabel("Nombre de Paris")
        self.ax.set_ylabel("Capital (€)")
        self.ax.legend()
        self.ax.grid(True, linestyle=":")
        self.canvas.draw()