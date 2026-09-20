import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Importations des algorithmes d'apprentissage automatique
from machine_learning.random_forest import simulation_ml_strategy
from machine_learning.grid_search_RF import run_grid_search

class OngletRF:
    """
    Classe gérant l'onglet 'Random Forest & Grid Search'.
    Permet d'évaluer le modèle de forêt aléatoire de manière simple (avec graphique d'évolution)
    ou d'exécuter une recherche exhaustive d'hyperparamètres (Grid Search) en affichant
    les logs en direct dans une console Tkinter intégrée.
    """
    def __init__(self, notebook, df, colors):
        self.notebook = notebook
        self.df = df
        self.COLORS = colors
        
        # Création et ajout de l'onglet
        self.frame = tk.Frame(self.notebook, bg=self.COLORS["bg_dark"])
        self.notebook.add(self.frame, text="Random Forest & Grid Search")
        
        self.creer_interface()
        
    def creer_interface(self):
        """
        Crée l'interface utilisateur pour l'onglet de Machine Learning.
        """
        panneau_gauche = tk.Frame(self.frame, padx=20, pady=20, width=320, bg=self.COLORS["bg_dark"])
        panneau_gauche.pack(side=tk.LEFT, fill=tk.Y)
        
        lbl_style = {"bg": self.COLORS["bg_dark"], "fg": self.COLORS["fg_text"]}
        entry_style = {"bg": self.COLORS["bg_card"], "fg": "white", "insertbackground": "white", "relief": "flat"}
        
        tk.Label(panneau_gauche, text="Machine Learning (RF)", font=("Segoe UI", 16, "bold"), **lbl_style).pack(pady=(0, 10))
        
        # Choix du mode : simple ou grid search
        self.mode_rf = tk.StringVar(value="simple")
        tk.Radiobutton(panneau_gauche, text="Mode : Simulation Simple (Graphique)", variable=self.mode_rf, value="simple", 
                       bg=self.COLORS["bg_dark"], fg=self.COLORS["accent"], selectcolor=self.COLORS["bg_card"], 
                       command=self.toggle_rf_view).pack(anchor="w")
        tk.Radiobutton(panneau_gauche, text="Mode : Grid Search (Console CSV)", variable=self.mode_rf, value="grid", 
                       bg=self.COLORS["bg_dark"], fg=self.COLORS["warning"], selectcolor=self.COLORS["bg_card"], 
                       command=self.toggle_rf_view).pack(anchor="w", pady=(0, 15))
        
        tk.Label(panneau_gauche, text="💡 Pour le Grid Search, séparez les\nvaleurs par des virgules (ex: 50, 100)", 
                 bg=self.COLORS["bg_dark"], fg=self.COLORS["fg_sub"], font=("Segoe UI", 8, "italic"), justify="left").pack(anchor="w", pady=(0,10))

        # Entrées des hyperparamètres classiques
        tk.Label(panneau_gauche, text="Capital Initial (€) :", **lbl_style).pack(anchor="w")
        self.entree_rf_capital = tk.Entry(panneau_gauche, **entry_style)
        self.entree_rf_capital.insert(0, "100")
        self.entree_rf_capital.pack(fill=tk.X, pady=(0, 5), ipady=3)
        
        tk.Label(panneau_gauche, text="Mise par pari (si fixe) (€) :", **lbl_style).pack(anchor="w")
        self.entree_rf_mise = tk.Entry(panneau_gauche, **entry_style)
        self.entree_rf_mise.insert(0, "1")
        self.entree_rf_mise.pack(fill=tk.X, pady=(0, 5), ipady=3)
        
        tk.Label(panneau_gauche, text="n_estimators (Arbres) :", **lbl_style).pack(anchor="w")
        self.entree_rf_nest = tk.Entry(panneau_gauche, **entry_style)
        self.entree_rf_nest.insert(0, "100")
        self.entree_rf_nest.pack(fill=tk.X, pady=(0, 5), ipady=3)
        
        tk.Label(panneau_gauche, text="max_depth (Profondeur) :", **lbl_style).pack(anchor="w")
        self.entree_rf_depth = tk.Entry(panneau_gauche, **entry_style)
        self.entree_rf_depth.insert(0, "5")
        self.entree_rf_depth.pack(fill=tk.X, pady=(0, 5), ipady=3)

        tk.Label(panneau_gauche, text="random_state (Seed) :", **lbl_style).pack(anchor="w")
        self.entree_rf_seed = tk.Entry(panneau_gauche, **entry_style)
        self.entree_rf_seed.insert(0, "42")
        self.entree_rf_seed.pack(fill=tk.X, pady=(0, 5), ipady=3)

        tk.Label(panneau_gauche, text="seuil_confiance (ex: 0.60) :", **lbl_style).pack(anchor="w")
        self.entree_rf_seuil = tk.Entry(panneau_gauche, **entry_style)
        self.entree_rf_seuil.insert(0, "0.0")
        self.entree_rf_seuil.pack(fill=tk.X, pady=(0, 10), ipady=3)

        # --- NOUVEAUX CHAMPS : KELLY CRITERION ---
        self.use_kelly_var = tk.BooleanVar(value=False)
        self.chk_kelly = tk.Checkbutton(panneau_gauche, text="Activer le Critère de Kelly", 
                                        variable=self.use_kelly_var, bg=self.COLORS["bg_dark"], 
                                        fg=self.COLORS["success"], selectcolor=self.COLORS["bg_card"], 
                                        activebackground=self.COLORS["bg_dark"], activeforeground="white",
                                        font=("Segoe UI", 9, "bold"))
        self.chk_kelly.pack(anchor="w", pady=(0, 5))

        tk.Label(panneau_gauche, text="Fraction de Kelly (ex: 0.25) :", **lbl_style).pack(anchor="w")
        self.entree_rf_kelly_frac = tk.Entry(panneau_gauche, **entry_style)
        self.entree_rf_kelly_frac.insert(0, "0.25")
        self.entree_rf_kelly_frac.pack(fill=tk.X, pady=(0, 20), ipady=3)
        # -----------------------------------------
        
        # Bouton d'action
        self.btn_lancer_rf = tk.Button(panneau_gauche, text="▶ LANCER RANDOM FOREST", 
                                       bg=self.COLORS["accent"], fg="white", font=("Segoe UI", 11, "bold"), 
                                       relief="flat", activebackground=self.COLORS["accent_hover"], 
                                       activeforeground="white", command=self.lancer_rf)
        self.btn_lancer_rf.pack(fill=tk.X, pady=5, ipady=5)

        self.lbl_rf_status = tk.Label(panneau_gauche, text="", font=("Segoe UI", 10, "italic"), 
                                       bg=self.COLORS["bg_dark"], fg=self.COLORS["warning"])
        self.lbl_rf_status.pack(fill=tk.X, pady=(5, 0))

        # Panneau d'affichage droit
        self.panneau_droit_rf = tk.Frame(self.frame, bg=self.COLORS["bg_dark"])
        self.panneau_droit_rf.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. Zone Graphique (Simulation Simple)
        self.frame_rf_graph = tk.Frame(self.panneau_droit_rf, bg="white")
        self.figure_rf = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax_rf = self.figure_rf.add_subplot(111)
        self.ax_rf.set_title("Prêt pour la simulation...")
        self.ax_rf.grid(True)
        self.canvas_rf = FigureCanvasTkAgg(self.figure_rf, master=self.frame_rf_graph)
        self.canvas_rf.draw()
        self.canvas_rf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 2. Zone Console (Grid Search)
        self.frame_rf_console = tk.Frame(self.panneau_droit_rf, bg=self.COLORS["bg_dark"])
        lbl_console = tk.Label(self.frame_rf_console, text="Console Grid Search", font=("Consolas", 12, "bold"), 
                               bg=self.COLORS["bg_dark"], fg=self.COLORS["warning"], anchor="w")
        lbl_console.pack(fill=tk.X, pady=5, padx=10)
        
        self.console_text = tk.Text(self.frame_rf_console, bg=self.COLORS["bg_card"], fg="#10b981", 
                                    font=("Consolas", 10), relief="flat", padx=10, pady=10)
        self.console_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Affichage par défaut (simulation simple)
        self.toggle_rf_view()

    def toggle_rf_view(self):
        """
        Bascule l'affichage à droite entre la console de Grid Search et le graphique.
        """
        mode = self.mode_rf.get()
        if mode == "simple":
            self.frame_rf_console.pack_forget()
            self.frame_rf_graph.pack(fill=tk.BOTH, expand=True)
            self.btn_lancer_rf.config(text="▶ LANCER SIMULATION", bg=self.COLORS["accent"])
        else:
            self.frame_rf_graph.pack_forget()
            self.frame_rf_console.pack(fill=tk.BOTH, expand=True)
            self.btn_lancer_rf.config(text="⚙️ LANCER GRID SEARCH", bg=self.COLORS["warning"])

    def log_console(self, msg):
        """
        Ajoute une ligne de texte dans la console Grid Search.
        """
        self.console_text.insert(tk.END, msg + "\n")
        self.console_text.see(tk.END)

    def lancer_rf(self):
        """
        Valide les entrées utilisateur et démarre l'évaluation du modèle (Simple ou Grid Search)
        dans un thread d'arrière-plan.
        """
        if self.df is None:
            messagebox.showerror("Erreur", "Données non chargées.")
            return

        mode = self.mode_rf.get()
        self.btn_lancer_rf.config(state=tk.DISABLED)
        
        try:
            capital_init = float(self.entree_rf_capital.get())
            mise_pari = float(self.entree_rf_mise.get())
            n_ests = [int(x.strip()) for x in self.entree_rf_nest.get().split(',')]
            depths = [int(x.strip()) for x in self.entree_rf_depth.get().split(',')]
            seeds = [int(x.strip()) for x in self.entree_rf_seed.get().split(',')]
            seuils = [float(x.strip()) for x in self.entree_rf_seuil.get().split(',')]
            
            # --- RÉCUPÉRATION DES PARAMÈTRES KELLY ---
            use_k = self.use_kelly_var.get()
            k_frac = float(self.entree_rf_kelly_frac.get())
            
        except ValueError:
            messagebox.showerror("Erreur de saisie", "Vérifiez vos paramètres. Utilisez uniquement des nombres (et des virgules pour le Grid Search).")
            self.btn_lancer_rf.config(state=tk.NORMAL)
            return

        def run_thread():
            if mode == "simple":
                self.frame.after(0, lambda: self.lbl_rf_status.config(text="Apprentissage RF en cours...", fg=self.COLORS["warning"]))
                try:
                    # ---> PASSAGE DES NOUVEAUX PARAMÈTRES ICI <---
                    res = simulation_ml_strategy(self.df, capital_initial=capital_init, mise=mise_pari, 
                                                 n_estimators=n_ests[0], max_depth=depths[0], 
                                                 random_state=seeds[0], seuil_confiance=seuils[0],
                                                 use_kelly=use_k, kelly_fraction=k_frac)
                    
                    self.frame.after(0, lambda: self.update_rf_graph(res, capital_init, n_ests[0], depths[0], seuils[0], use_k))
                    self.frame.after(0, lambda: self.lbl_rf_status.config(text="Simulation terminée ✅", fg=self.COLORS["success"]))
                except Exception as e:
                    self.frame.after(0, lambda: messagebox.showerror("Erreur", f"Erreur IA : {e}"))
                    self.frame.after(0, lambda: self.lbl_rf_status.config(text="Erreur ❌", fg=self.COLORS["danger"]))

            elif mode == "grid":
                self.frame.after(0, lambda: self.lbl_rf_status.config(text="Grid Search en cours...", fg=self.COLORS["warning"]))
                self.frame.after(0, lambda: self.console_text.delete(1.0, tk.END))
                
                # Callback pour rediriger les sorties log de grid_search_RF vers l'UI
                def tkinter_logger(msg):
                    self.frame.after(0, lambda m=msg: self.log_console(m))

                # Note: Le grid search de base du projet ne gère pas encore l'injection de Kelly dans la boucle.
                # Il utilisera les paramètres de mise fixe par défaut.
                df_res = run_grid_search(
                    df=self.df,
                    list_n_estimators=n_ests,
                    list_max_depth=depths,
                    list_random_state=seeds,
                    list_seuil_confiance=seuils,
                    fichier_sortie="resultats_grid_search_interface.csv",
                    logger=tkinter_logger,
                    capital_initial=capital_init,
                    mise=mise_pari,
                    use_kelly=use_k,          
                    kelly_fraction=k_frac
                )
                
                if not df_res.empty:
                    self.frame.after(0, lambda: self.log_console("\n🏆 TOP 10 DES MEILLEURES COMBINAISONS (Le reste est dans le CSV) :"))
                    top10 = df_res.head(10)
                    for idx, row in top10.iterrows():
                        txt = f"ROI: {row['roi']}% | Cap: {row['capital_final']}€ | Paris: {row['nb_paris']} | (n_est={row['n_estimators']}, depth={row['max_depth']}, seuil={row['seuil_confiance']})"
                        self.frame.after(0, lambda m=txt: self.log_console(m))

                self.frame.after(0, lambda: self.lbl_rf_status.config(text="Grid Search terminé ✅", fg=self.COLORS["success"]))

            # Réactivation du bouton
            self.frame.after(0, lambda: self.btn_lancer_rf.config(state=tk.NORMAL))

        threading.Thread(target=run_thread, daemon=True).start()

    def update_rf_graph(self, resultats, capital_init, n, d, seuil, use_k):
        """
        Dessine la courbe de capital résultant de la simulation simple sur le graphique droit.
        """
        self.ax_rf.clear()
        
        # Changement de couleur dynamique si Kelly est activé
        couleur_courbe = "purple" if use_k else "blue"
        label_courbe = "Capital RF (Kelly)" if use_k else "Capital RF (Mise Fixe)"
        
        self.ax_rf.plot(resultats['historique'], label=label_courbe, color=couleur_courbe, linewidth=2)
        self.ax_rf.axhline(y=capital_init, color='r', linestyle='--', label=f'Départ ({capital_init}€)')
        
        titre_mode = "AVEC Kelly" if use_k else "SANS Kelly"
        self.ax_rf.set_title(f"Random Forest {titre_mode} (n={n}, depth={d}, seuil={seuil})\nROI: {resultats['roi']}% | Paris: {resultats['nb_paris']} | WR: {resultats['win_rate']}% | Capital Final : {resultats['capital_final']}€", fontdict={'weight': 'bold'})
        
        self.ax_rf.set_xlabel("Nombre de paris")
        self.ax_rf.set_ylabel("Capital (€)")
        self.ax_rf.legend()
        self.ax_rf.grid(True, linestyle=":")
        self.canvas_rf.draw()