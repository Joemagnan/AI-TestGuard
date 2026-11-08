# Rapport d'évaluation du modèle

Données : 284 807 transactions, 1 081 doublons exacts supprimés (19 fraudes), soit 283 726 transactions dont 473 fraudes (0.167%).
Découpage stratifié 60/20/20 : entraînement 170235 (284 fraudes), validation 56745 (94), test 56746 (95).

## Protocole

1. Hyperparamètres choisis sur la validation (PR-AUC), grille réduite par famille.
2. Seuil choisi sur la validation : coût attendu minimal (25 x FN + 1 x FP) sous la contrainte FPR <= 0.5%.
3. Modèle retenu : coût attendu minimal sur la validation.
4. Évaluation unique sur le test, avec des critères fixés avant (commit 6064d6f).

## Comparaison des modèles

| Modèle | Jeu | Seuil | VP | FP | FN | VN | Rappel | Précision | F1 | FPR | FNR | PR-AUC | ROC-AUC | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Régression logistique | validation | 0.020 | 84 | 55 | 10 | 56596 | 0.894 | 0.604 | 0.721 | 9.7e-04 | 0.106 | 0.818 | 0.974 | 0.999 |
| Arbre de décision | validation | 0.148 | 83 | 29 | 11 | 56622 | 0.883 | 0.741 | 0.806 | 5.1e-04 | 0.117 | 0.849 | 0.945 | 0.999 |
| Régression logistique | test | 0.020 | 75 | 53 | 20 | 56598 | 0.789 | 0.586 | 0.673 | 9.4e-04 | 0.211 | 0.680 | 0.946 | 0.999 |
| Arbre de décision | test | 0.148 | 72 | 33 | 23 | 56618 | 0.758 | 0.686 | 0.720 | 5.8e-04 | 0.242 | 0.695 | 0.892 | 0.999 |

Modèle retenu : **Arbre de décision** (seuil 0.1481). Sur la validation, les deux modèles sont presque à égalité de coût (304 contre 305) : le choix suit la règle fixée à l'avance mais n'est pas un écart significatif.

## Évaluation finale (test) et incertitude

| Métrique | Valeur | IC 95 % (bootstrap stratifié, 1 000 tirages) |
|---|---|---|
| recall | 0.758 | [0.674 ; 0.842] |
| precision | 0.686 | [0.610 ; 0.775] |
| fpr | 5.8e-04 | [0.000 ; 0.001] |
| pr_auc | 0.695 | [0.602 ; 0.792] |

| Critère d'acceptation | Valeur test | Exigence | Résultat |
|---|---|---|---|
| recall | 0.758 | >= 0.75 | atteint |
| precision | 0.686 | >= 0.2 | atteint |
| fpr | 5.8e-04 | <= 0.005 | atteint |
| pr_auc | 0.695 | >= 0.6 | atteint |

## Pourquoi l'accuracy ne suffit pas

Un modèle qui déclare toutes les transactions normales obtient une accuracy de 99.83% sur le test, sans détecter une seule fraude (rappel 0). Le modèle retenu a une accuracy de 99.90% : l'écart réel se lit sur le rappel, la précision et la PR-AUC.

## Surapprentissage (entraînement / validation / test, seuil retenu)

| Jeu | Rappel | Précision | FPR | PR-AUC |
|---|---|---|---|---|
| entraînement | 0.842 | 0.679 | 6.6e-04 | 0.827 |
| validation | 0.883 | 0.741 | 5.1e-04 | 0.849 |
| test | 0.758 | 0.686 | 5.8e-04 | 0.695 |

## Recherche d'hyperparamètres (validation)

| Modèle | Paramètres | PR-AUC validation | ROC-AUC validation |
|---|---|---|---|
| Arbre de décision | {"max_depth": 8, "min_samples_leaf": 20, "class_weight": null} | 0.849 | 0.945 |
| Arbre de décision | {"max_depth": 6, "min_samples_leaf": 20, "class_weight": null} | 0.846 | 0.945 |
| Arbre de décision | {"max_depth": 4, "min_samples_leaf": 20, "class_weight": null} | 0.832 | 0.941 |
| Arbre de décision | {"max_depth": 4, "min_samples_leaf": 1, "class_weight": null} | 0.829 | 0.931 |
| Arbre de décision | {"max_depth": 6, "min_samples_leaf": 1, "class_weight": null} | 0.779 | 0.882 |
| Arbre de décision | {"max_depth": 8, "min_samples_leaf": 20, "class_weight": "balanced"} | 0.703 | 0.925 |
| Arbre de décision | {"max_depth": 8, "min_samples_leaf": 1, "class_weight": null} | 0.697 | 0.818 |
| Arbre de décision | {"max_depth": 8, "min_samples_leaf": 1, "class_weight": "balanced"} | 0.493 | 0.924 |
| Arbre de décision | {"max_depth": 6, "min_samples_leaf": 20, "class_weight": "balanced"} | 0.483 | 0.933 |
| Arbre de décision | {"max_depth": 6, "min_samples_leaf": 1, "class_weight": "balanced"} | 0.479 | 0.933 |
| Arbre de décision | {"max_depth": 4, "min_samples_leaf": 1, "class_weight": "balanced"} | 0.093 | 0.940 |
| Arbre de décision | {"max_depth": 4, "min_samples_leaf": 20, "class_weight": "balanced"} | 0.093 | 0.940 |
| Régression logistique | {"C": 0.1, "class_weight": null} | 0.818 | 0.974 |
| Régression logistique | {"C": 0.01, "class_weight": null} | 0.818 | 0.978 |
| Régression logistique | {"C": 1.0, "class_weight": null} | 0.818 | 0.973 |
| Régression logistique | {"C": 0.1, "class_weight": "balanced"} | 0.813 | 0.980 |
| Régression logistique | {"C": 1.0, "class_weight": "balanced"} | 0.813 | 0.980 |
| Régression logistique | {"C": 0.01, "class_weight": "balanced"} | 0.808 | 0.980 |
