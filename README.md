# ⚽ Simulation and Evaluation of Sports Betting Strategies


Projet académique de Master 1 Informatique (Université de Caen Normandie) visant à modéliser, backtester et évaluer la viabilité financière de différentes stratégies de paris sportifs sur la Premier League anglaise face à la marge structurelle des bookmakers (*overround*).

---

### 🌟 Fonctionnalités Clés

- **Backtesting Multi-Stratégies :** Évaluation empirique d'approches naïves (Home/Away/Draw), de suivi du risque (Low/High risk) et d'heuristiques de *Value Betting*.
- **Machine Learning Supervisé :** Entraînement d'un classifieur **Random Forest** avec split temporel strict.
- **Filtrage par Seuil de Confiance ($S_c$) :** Mécanisme d'abstention décisionnelle pour isoler les signaux à haute probabilité.
- **Optimisation Combinatoire :** Module de **Grid Search** multivariable pour cartographier le compromis volume / précision.
- **Implémentation du Critère de Kelly Fractionnaire** pour dynamiser les mises selon l'avantage mathématique.
- **Intelligence Contextuelle (NLP) :** Pipeline d'analyse de sentiment de presse sportive via *TextBlob* et *Newspaper3k*.
- **Interface Graphique Complète :** GUI modulaire développée sous *Tkinter* (visualisation en direct via *Matplotlib*, console d'exécution et simulateur H2H unitaire).
