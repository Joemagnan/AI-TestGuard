# Surveillance simulée (6 lots de production tirés du jeu de test)

Taux de transactions signalées attendu (validation) : 0.197%. Variables critiques surveillées : V17, V10, V14. Script : `scripts/simulate_monitoring.py`.

## Constat préalable : cycle jour/nuit

Découpé en tranches chronologiques de 8 h, le jeu de test SANS incident déclenche des alertes de dérive : Time change évidemment de distribution, et certaines composantes suivent le rythme jour/nuit. Comparer une tranche horaire au profil d'entraînement (48 h) produit donc de fausses alertes : la surveillance doit comparer des fenêtres de 24 h, ou une tranche à la même tranche horaire de référence, et exclure Time du calcul de dérive.

| Tranche | Heures | PSI Time | Autre variable la plus dérivante | PSI | Statut |
|---|---|---|---|---|---|
| 1 | 0.0 h - 12.0 h | 10.529 | V1 | 2.412 | alerte |
| 2 | 12.0 h - 18.0 h | 9.105 | V1 | 2.395 | alerte |
| 3 | 18.0 h - 23.5 h | 10.513 | V1 | 2.407 | alerte |
| 4 | 23.5 h - 35.6 h | 9.423 | V1 | 0.355 | alerte |
| 5 | 35.6 h - 41.3 h | 9.105 | V1 | 0.484 | alerte |
| 6 | 41.3 h - 48.0 h | 10.517 | V1 | 0.518 | alerte |

## Indicateurs disponibles immédiatement (sans labels)

| Fenêtre | Incident injecté | Taux signalé | Ratio / attendu | Dérive (PSI max) | Lignes incomplètes | Alertes émises |
|---|---|---|---|---|---|---|
| 1 | aucun | 0.180% | 0.911 | stable (0.00) | 0.0% | aucune |
| 2 | aucun | 0.211% | 1.071 | stable (0.00) | 0.0% | aucune |
| 3 | aucun | 0.159% | 0.804 | stable (0.00) | 0.0% | aucune |
| 4 | dérive des données : montants x 3 | 0.116% | 0.589 | alerte (0.41) | 0.0% | dérive alerte (Amount, V7) |
| 5 | panne amont : V17, V10, V14 absents (5 % des lignes) | 0.211% | 1.071 | stable (0.00) | 4.9% | données manquantes 4.9%; 464 revues manuelles |
| 6 | dérive de concept : fraudes déguisées | 0.085% | 0.429 | stable (0.00) | 0.0% | taux signalé x 0.43 |

## Indicateurs disponibles à l'arrivée des labels (contestations, enquêtes)

| Fenêtre | Fraudes réelles | Rappel | Précision | FPR |
|---|---|---|---|---|
| 1 | 19 | 0.684 | 0.765 | 4.2e-04 |
| 2 | 18 | 0.833 | 0.750 | 5.3e-04 |
| 3 | 8 | 0.875 | 0.467 | 8.5e-04 |
| 4 | 10 | 0.700 | 0.636 | 4.2e-04 |
| 5 | 25 | 0.640 | 0.800 | 4.2e-04 |
| 6 | 15 | 0.000 | 0.000 | 8.5e-04 |

Lecture : la dérive des données (F4) et la panne amont (F5) sont visibles dès l'arrivée des transactions. La dérive de concept (F6) ne modifie presque pas la distribution des entrées (les fraudes représentent 0,17 % du flux) : seule une baisse du taux de transactions signalées peut l'annoncer, et elle n'est confirmée qu'à l'arrivée des labels, par la chute du rappel.
