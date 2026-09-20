import pandas as pd
import time
from machine_learning.random_forest import simulation_ml_strategy

def run_grid_search(df, list_n_estimators, list_max_depth, list_random_state, list_seuil_confiance, fichier_sortie="resultats_grid_search.csv", logger=print,capital_initial=100, mise=1, use_kelly=False, kelly_fraction=0.25):
    """
    Teste toutes les combinaisons possibles des paramètres et sauvegarde les res
    """
    resultats_finaux = []
    
    total_iterations = len(list_n_estimators) * len(list_max_depth) * len(list_random_state) * len(list_seuil_confiance)
    iteration_actuelle = 0
    
    logger(f"Début du Grid Search : {total_iterations} combinaisons à tester.")
    start_time = time.time()

    for n_est in list_n_estimators:
        for m_depth in list_max_depth:
            for r_state in list_random_state:
                for seuil in list_seuil_confiance:
                    iteration_actuelle += 1
                    
                    logger(f"[{iteration_actuelle}/{total_iterations}] Test avec : n_est={n_est}, depth={m_depth}, seed={r_state}, seuil={seuil}")
                    
                    try:
                        res = simulation_ml_strategy(
                            df_original=df, 
                            capital_initial=capital_initial, 
                            mise=mise, 
                            n_estimators=n_est, 
                            max_depth=m_depth, 
                            random_state=r_state,
                            seuil_confiance=seuil,
                            use_kelly = use_kelly,    
                            kelly_fraction = kelly_fraction
                        )
                        
                        resultats_finaux.append({
                            'n_estimators': n_est,
                            'max_depth': m_depth,
                            'random_state': r_state,
                            'seuil_confiance': seuil,
                            'nb_paris': res['nb_paris'],
                            'win_rate': res['win_rate'],
                            'profit': res['profit'],
                            'roi': res['roi'],
                            'capital_final': res['capital_final']
                        })
                        
                        logger(f"  -> Fini ! Paris: {res['nb_paris']} | ROI: {res['roi']}% | Capital: {res['capital_final']}€")
                        
                    except ZeroDivisionError:
                        logger(" Aucun pari placé avec ces paramètres (seuil trop grand).")
                        resultats_finaux.append({
                            'n_estimators': n_est,
                            'max_depth': m_depth,
                            'random_state': r_state,
                            'seuil_confiance': seuil,
                            'nb_paris': 0,
                            'win_rate': 0,
                            'profit': 0,
                            'roi': 0,
                            'capital_final': 100
                        })

    df_resultats = pd.DataFrame(resultats_finaux)
    
    # Sauvegarde en CSV
    df_resultats = df_resultats.sort_values(by="capital_final", ascending = False)
    df_resultats.to_csv(fichier_sortie, index=False)
    
    temps_ecoule = round(time.time() - start_time, 2)
    logger(f"\n Grid Search terminé en {temps_ecoule} secondes")
    logger(f" les résultats ont été sauvegardés dans '{fichier_sortie}'")
    
    return df_resultats

if __name__ == "__main__":
    df = pd.read_csv('../charger_data/epl_historique.csv')

    grille_n_estimators = [50, 100]
    grille_max_depth = [2, 3]
    grille_random_state = [42] 
    grille_seuil = [0.0, 0.60]
    
    df_res = run_grid_search(
        df=df,
        list_n_estimators=grille_n_estimators,
        list_max_depth=grille_max_depth,
        list_random_state=grille_random_state,
        list_seuil_confiance=grille_seuil,
        fichier_sortie="mes_resultats_rf.csv"
    )