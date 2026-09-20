import tkinter as tk
from tkinter import ttk
from charger_data.charger_donnees import charger_donnees
from charger_data.data_manager import DataManager

# Importations des onglets modularisés
from onglets.onglet_simulation import OngletSimulation
from onglets.onglet_cotes import OngletCotes
from onglets.onglet_rf import OngletRF
from onglets.onglet_predictions import OngletPredictions
from onglets.onglet_h2h import OngletH2H
from onglets.onglet_value_betting import OngletValueBetting

class ApplicationParis(tk.Tk):
    """
    Classe principale de l'application de simulation de paris sportifs.
    Configure la fenêtre principale, gère les couleurs de thèmes communes, configure
    les styles personnalisés des composants Tkinter, charge les données historiques
    et instancie les différents onglets de l'application dans un Notebook.
    """
    def __init__(self):
        super().__init__()
        self.title("Simulateur de Paris Sportifs - Projet Python")
        self.geometry("1400x850") 
        
        # --- THEME COLORS (Slate & Indigo) ---
        self.COLORS = {
            "bg_dark": "#020617",         
            "bg_card": "#1e293b",         
            "fg_text": "#e2e8f0",         
            "fg_sub": "#94a3b8",          
            "accent": "#6366f1",          
            "accent_hover": "#4f46e5",    
            "success": "#22c55e",         
            "danger": "#ef4444",          
            "warning": "#f59e0b",         
            "border": "#334155"           
        }
        
        self.configure(bg=self.COLORS["bg_dark"])
        self.configure_styles()
        
        # Initialisation des services de données
        self.data_manager = DataManager(update_callback=self.on_data_update)
        self.df = charger_donnees("charger_data/epl_historique.csv")
        
        # Création de l'interface par onglets
        self.creer_notebook()
        
        # Charger automatiquement les cotes live au démarrage après 500 ms
        self.after(500, lambda: self.onglet_cotes.charger_cotes_live())
        
    def on_data_update(self, event_type, data):
        """
        Callback de mise à jour des données (actuellement pass-through).
        """
        pass
        
    def configure_styles(self):
        """
        Configure les styles TTK pour correspondre à notre charte graphique moderne.
        """
        style = ttk.Style()
        style.theme_use('clam') 
        
        c = self.COLORS
        
        # Configuration des polices et fonds d'écran par défaut
        style.configure(".", background=c["bg_dark"], foreground=c["fg_text"], fieldbackground=c["bg_card"])
        style.configure("TFrame", background=c["bg_dark"])
        style.configure("Card.TFrame", background=c["bg_card"], relief="flat")
        
        # Style des Onglets (Notebook)
        style.configure("TNotebook", background=c["bg_dark"], borderwidth=0)
        style.configure("TNotebook.Tab", background=c["bg_card"], foreground=c["fg_sub"], 
                        padding=[15, 8], font=("Segoe UI", 10, "bold"), borderwidth=0)
        style.map("TNotebook.Tab", 
                  background=[("selected", c["accent"]), ("active", c["accent_hover"])], 
                  foreground=[("selected", "white"), ("active", "white")])
        
        # Style des Tableaux (Treeview)
        style.configure("Treeview", background=c["bg_card"], foreground=c["fg_text"], 
                        fieldbackground=c["bg_card"], font=("Segoe UI", 9), rowheight=35, borderwidth=0)
        style.configure("Treeview.Heading", background=c["bg_dark"], foreground=c["fg_sub"], 
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("Treeview", 
                  background=[("selected", c["accent_hover"])], 
                  foreground=[("selected", "white")])
                  
        # Barre de défilement (Scrollbar)
        style.configure("Vertical.TScrollbar", background=c["bg_card"], 
                        troughcolor=c["bg_dark"], bordercolor=c["bg_dark"], arrowcolor=c["fg_sub"])
        # Pour regler le probleme du blanc sur blanc sur les predictions
        style.configure("TCombobox", 
                        fieldbackground=c["bg_card"], 
                        background=c["bg_dark"], 
                        foreground=c["fg_text"])
        style.map("TCombobox", 
                  fieldbackground=[("readonly", c["bg_card"]), ("focus", c["accent"])],
                  foreground=[("readonly", c["fg_text"]), ("focus", "white")],
                  selectbackground=[("focus", c["accent"])],
                  selectforeground=[("focus", "white")])

    def creer_notebook(self):
        """
        Instancie et ajoute chaque onglet de l'application dans le Notebook principal.
        """
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Onglet 1 : Simulation Stratégies (Heuristiques)
        self.onglet_sim = OngletSimulation(self.notebook, self.df, self.COLORS)
        
        # Onglet 2 : Cotes & Scores Impliqués (Winamax / Betclic)
        self.onglet_cotes = OngletCotes(self.notebook, self.data_manager, self.COLORS)
        
        # Onglet 3 : Random Forest & Grid Search (Machine Learning)
        self.onglet_rf = OngletRF(self.notebook, self.df, self.COLORS)
        
        # Onglet 4 : Prédictions (News & Sentiment)
        self.onglet_pred = OngletPredictions(self.notebook, self.COLORS)

        self.onglet_h2h = OngletH2H(self.notebook, self.df, self.COLORS)
        
        # Onglet 6 : Value Betting (Forme Historique)
        self.onglet_value_betting = OngletValueBetting(self.notebook, self.df, self.COLORS)
        
if __name__ == "__main__":
    app = ApplicationParis()
    app.mainloop()