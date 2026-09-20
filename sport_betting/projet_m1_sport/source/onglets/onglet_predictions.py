import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import os
import threading

# Importation du script externe de prédiction (Web scraping / NLP)
from informations_and_heuristic_model.analyse_article import run_prediction_pipeline

class OngletPredictions:
    """
    Classe gérant l'onglet 'Prédictions (News & Sentiment)'.
    Permet de déclencher un script de web scraping et d'analyse de sentiment (NLP)
    pour prédire l'issue des prochains matchs à J+N jours, et d'importer le CSV 
    généré dans un tableau d'analyse.
    """
    def __init__(self, notebook, colors):
        self.notebook = notebook
        self.COLORS = colors
        
        # Création et ajout de l'onglet
        self.frame = tk.Frame(self.notebook, bg=self.COLORS["bg_dark"])
        self.notebook.add(self.frame, text="Prédictions (News & Sentiment)")
        
        self.creer_interface()
        
    def creer_interface(self):
        """
        Crée la barre d'outils et le tableau Treeview de l'onglet.
        """
        # Barre de contrôles supérieure
        toolbar = tk.Frame(self.frame, bg=self.COLORS["bg_dark"], pady=20, padx=20)
        toolbar.pack(fill=tk.X)
        
        tk.Label(toolbar, text="Jours après (+):", bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_text"]).pack(side=tk.LEFT, padx=5)
        self.entree_jours_pred = tk.Entry(toolbar, width=5, bg=self.COLORS["bg_card"], fg="white", 
                                           insertbackground="white", relief="flat")
        self.entree_jours_pred.insert(0, "1") 
        self.entree_jours_pred.pack(side=tk.LEFT, padx=5)

        tk.Button(toolbar, text="🔄 GÉNÉRER PRÉDICTIONS", bg=self.COLORS["accent"], fg="white", 
                  font=("Segoe UI", 10, "bold"), command=self.lancer_script_prediction, relief="flat", padx=15).pack(side=tk.LEFT, padx=5)
        
        tk.Button(toolbar, text="📂 CHARGER CSV", bg=self.COLORS["bg_card"], fg=self.COLORS["fg_text"], 
                  command=self.charger_csv_predictions, relief="flat", padx=15).pack(side=tk.LEFT, padx=5)

        self.lbl_status_pred = tk.Label(toolbar, text="Prêt", bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_sub"])
        self.lbl_status_pred.pack(side=tk.LEFT, padx=20)

        # Tableau des résultats de sentiment/décisions
        cols = ("Match", "Cote H", "Cote N", "Cote A", "Sent. Home", "Sent. Away", "DÉCISION")
        self.tree_pred = ttk.Treeview(self.frame, columns=cols, show='headings')
        
        for col in cols:
            self.tree_pred.heading(col, text=col)
            self.tree_pred.column(col, anchor="center", width=120)
        
        self.tree_pred.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

    def lancer_script_prediction(self):
        """
        Déclenche l'exécution du pipeline de web scraping et analyse NLP dans un thread séparé.
        """
        try:
            jours = int(self.entree_jours_pred.get())
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre de jours valide.")
            return

        self.lbl_status_pred.config(text=f"Analyse pour J+{jours} en cours...", fg=self.COLORS["warning"])
        
        def run():
            try:
                nb_matchs = run_prediction_pipeline(days_ahead=jours) 
                if nb_matchs > 0:
                    self.frame.after(0, lambda: self.lbl_status_pred.config(text=f"✅ {nb_matchs} matchs (J+{jours})", fg=self.COLORS["success"]))
                    messagebox.showinfo("Succès", f"Fichier généré pour J+{jours} !")
                else:
                    self.frame.after(0, lambda: self.lbl_status_pred.config(text="ℹ️ Aucun match trouvé", fg=self.COLORS["fg_sub"]))
                    messagebox.showwarning("Info", f"Aucun match trouvé pour J+{jours}.")
            except Exception as e:
                self.frame.after(0, lambda: messagebox.showerror("Erreur", f"Erreur : {e}"))

        threading.Thread(target=run, daemon=True).start()

    def charger_csv_predictions(self):
        """
        Ouvre un explorateur de fichiers pour charger les prédictions depuis un fichier CSV et alimente le tableau.
        """
        filepath = filedialog.askopenfilename(title="Ouvrir les prédictions", filetypes=[("CSV files", "*.csv")])
        if not filepath: 
            return
            
        try:
            df_pred = pd.read_csv(filepath)
            for item in self.tree_pred.get_children(): 
                self.tree_pred.delete(item)
                
            for _, row in df_pred.iterrows():
                self.tree_pred.insert("", "end", values=(
                    f"{row['HomeTeam']} vs {row['AwayTeam']}", 
                    row['Cote_H'], row['Cote_N'], row['Cote_A'],
                    row['Sent_Home'], row['Sent_Away'], row['Decision']
                ))
            self.lbl_status_pred.config(text=f"Chargé : {os.path.basename(filepath)}", fg=self.COLORS["success"])
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de lire le fichier : {e}")
