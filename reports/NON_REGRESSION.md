# Rapport de non-régression

Référence figée : version 1.0.0 (commit 6064d6f), seuil 0.1481. Toutes les versions sont comparées sur le **même jeu de validation fixe** (empreinte vérifiée) ; le jeu de test n'est pas utilisé.

Règle : dégradation > tolérance sur une métrique -> **rejeter** ; > moitié de la tolérance -> **investiguer** ; sinon -> **accepter**. Tolérances : pr_auc 0.02, recall 0.03, precision 0.05, fpr 0.001.

| Version | Changement | Seuil | Rappel | Précision | FPR | PR-AUC | Décision |
|---|---|---|---|---|---|---|---|
| 1.0.0 (référence) | - | 0.148 | 0.883 | 0.741 | 5.1e-04 | 0.849 | - |
| 1.1.0-a | Time encodé en heure de la journée (sin, cos) au lieu des secondes brutes | 0.179 | 0.883 | 0.806 | 3.5e-04 | 0.842 | ACCEPTER |
| 1.1.0-b | Régression logistique (C = 0,1), deuxième modèle de la comparaison | 0.020 | 0.894 | 0.604 | 9.7e-04 | 0.818 | REJETER |
| 1.1.0-c | Arbre moins profond (profondeur 3) pour simplifier le modèle | 0.178 | 0.883 | 0.629 | 8.6e-04 | 0.768 | REJETER |
| 1.1.0-d | Règle de seuil modifiée : coût FN = coût FP (au lieu de 25 pour 1) | 0.636 | 0.830 | 0.886 | 1.8e-04 | 0.849 | REJETER |

## Motifs

- **1.1.0-a** (accepter) : aucune dégradation au-delà de la moitié des tolérances
- **1.1.0-b** (rejeter) : pr_auc : dégradation de 0.0305 (tolérance 0.02). ; precision : dégradation de 0.1368 (tolérance 0.05).
- **1.1.0-c** (rejeter) : pr_auc : dégradation de 0.0805 (tolérance 0.02). ; precision : dégradation de 0.1123 (tolérance 0.05).
- **1.1.0-d** (rejeter) : recall : dégradation de 0.0532 (tolérance 0.03).
