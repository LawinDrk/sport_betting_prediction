
def calcul_resultats(cap_init, cap_final, nb_paris, nb_victoires, historique, mise = 1):
    profit = cap_final - cap_init
    roi = (profit / (nb_paris * mise)) * 100 if nb_paris > 0 else 0 
    win_rate = (nb_victoires / nb_paris) * 100 if nb_paris > 0 else 0
    
    return {
        "capital_final": round(cap_final, 2),
        "profit": round(profit, 2),
        "roi": round(roi, 2),
        "nb_paris": nb_paris,
        "win_rate": round(win_rate, 2),
        "historique": historique
    }