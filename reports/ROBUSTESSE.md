# Rapport de robustesse et de sensibilité

Modèle livré v1.0.0, jeu de test complet (56 746 transactions, 95 fraudes), seuil 0.1481. Référence : rappel 0.758, précision 0.686, FPR 0.00058. Script : `scripts/run_robustness.py`.

## 1. Valeurs manquantes (10 % des lignes) - comportement : traiter + alerte

| Scénario | Fraudes touchées | Rappel avant (touchées) | Rappel après (touchées) | Δ rappel global | Δ FPR global | Alerte |
|---|---|---|---|---|---|---|
| V1 manquante | 10 | 1.000 | 1.000 | 0.000 | 0.000 | oui |
| 3 variables les plus importantes (V17, V10, V14) | 10 | 1.000 | 0.000 | -0.105 | -3.5e-05 | oui |
| V1 à V28 manquantes | 10 | 1.000 | 0.000 | -0.105 | -3.5e-05 | oui |
| Amount manquant | 10 | 1.000 | 1.000 | 0.000 | 0.000 | oui |

## 2. Valeurs infinies et non numériques

- Infini traité exactement comme une valeur manquante : True
- Lot contenant une valeur texte rejeté : True

## 3. Montants extrêmes - comportement : traiter + alerte

| Scénario | Probabilités valides | Décisions changées | Taux d'alerte | Δ rappel | Δ FPR |
|---|---|---|---|---|---|
| Amount x 10 | oui | 0.000 | 0.002 | 0.000 | 0.000 |
| Amount x 100 | oui | 0.000 | 0.106 | 0.000 | 0.000 |
| Amount = 1 000 000 | oui | 0.000 | 1.000 | 0.000 | 0.000 |
| Amount = 1e12 | oui | 0.000 | 1.000 | 0.000 | 0.000 |

## 4. Perturbations limitées - sensibilité mesurée (moyenne sur 5 tirages)

Bruit gaussien sur V1 à V28 proportionnel à l'écart-type d'entraînement de chaque composante ; Amount et Time sont perturbés séparément (amplitudes propres).

| Amplitude (x écart-type) | |Δ proba| moyen | Décisions changées | Fraudes changées | Changées près du seuil | Rappel | Précision | FPR |
|---|---|---|---|---|---|---|---|
| 0.010 | 1.1e-05 | 0.000 | 0.000 | 0.000 | 0.758 | 0.686 | 5.8e-04 |
| 0.050 | 4.6e-05 | 3.5e-05 | 0.000 | 0.014 | 0.758 | 0.681 | 6.0e-04 |
| 0.100 | 7.4e-05 | 6.3e-05 | 0.002 | 0.024 | 0.756 | 0.676 | 6.1e-04 |
| 0.250 | 1.7e-04 | 1.4e-04 | 0.017 | 0.029 | 0.762 | 0.666 | 6.4e-04 |
| 0.500 | 4.0e-04 | 3.6e-04 | 0.029 | 0.043 | 0.771 | 0.611 | 8.3e-04 |

| Perturbation | |Δ proba| moyen | Décisions changées | Δ rappel | Δ précision | Δ FPR |
|---|---|---|---|---|---|
| Amount +/- 1 % (multiplicatif) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Amount +/- 5 % (multiplicatif) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Time +/- 60 s | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Time +/- 1 h | 2.6e-06 | 1.8e-05 | 0.000 | -0.006 | 1.8e-05 |

## 5. Distribution différente - comportement : alerte de dérive (PSI)

| Scénario | Statut dérive | PSI max | Variables en tête | Taux signalé | Δ rappel | Δ FPR |
|---|---|---|---|---|---|---|
| Jeu de test sans modification (contrôle) | stable | 4.7e-04 | V16, V14, V11 | 0.002 | 0.000 | 0.000 |
| Montants x 3 (inflation, nouveau commerce) | alerte | 0.410 | Amount, V16, V14 | 0.002 | 0.000 | 0.000 |
| Décalage +0,5 écart-type sur V17, V10, V14 | alerte | 0.521 | V10, V17, V14 | 0.002 | -0.032 | -1.8e-04 |
| Échelle x 1,5 sur V1 à V28 | alerte | 0.481 | V1, V22, V24 | 0.007 | 0.042 | 0.005 |

Prévalence multipliée par 5 (taux de fraude 0.831%) : rappel 0.758, précision 0.916, FPR 0.00058.

### Validation temporelle (passé -> futur)

Entraînement sur [0 ; 100 000[ s, seuil sur [100 000 ; 130 000[ s, test sur [130 000 ; 172 792] s (110 fraudes) : rappel 0.727, précision 0.708, FPR 0.00036, PR-AUC 0.673.

### Variabilité d'échantillonnage (5 plis, seuil livré fixe, sans le jeu de test)

| Pli | Fraudes | Rappel | Précision | FPR | PR-AUC |
|---|---|---|---|---|---|
| 1 | 75 | 0.787 | 0.728 | 4.9e-04 | 0.769 |
| 2 | 75 | 0.747 | 0.848 | 2.2e-04 | 0.736 |
| 3 | 76 | 0.855 | 0.722 | 5.5e-04 | 0.801 |
| 4 | 76 | 0.842 | 0.736 | 5.1e-04 | 0.804 |
| 5 | 76 | 0.776 | 0.694 | 5.7e-04 | 0.765 |

## 6. Schéma invalide - comportement : rejeter

| Cas | Résultat |
|---|---|
| colonne Amount absente | rejeté : Colonnes obligatoires absentes : ['Amount']. |
| colonne V1 renommée v1 | rejeté : Colonnes obligatoires absentes : ['V1']. |
| colonne V2 dupliquée | rejeté : Colonnes dupliquées : ['V2']. |
| V5 de type texte | rejeté : Type incorrect (numérique attendu) : ['V5']. |
| lot vide | rejeté : Le jeu de données est vide. |
| montant négatif | rejeté : Valeurs négatives interdites dans « Amount ». |
| colonne supplémentaire (card_id) | accepté (colonne ignorée et journalisée) |

## Variables les plus importantes du modèle livré

| Variable | Importance |
|---|---|
| V17 | 0.728 |
| V10 | 0.097 |
| V14 | 0.091 |
| Time | 0.044 |
| V26 | 0.031 |
| V4 | 0.006 |
| V9 | 0.002 |
| V7 | 7.3e-04 |
| V1 | 0.000 |
| V2 | 0.000 |
