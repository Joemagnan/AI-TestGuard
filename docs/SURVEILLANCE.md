# Stratégie de surveillance après déploiement

## Deux dérives à distinguer

- **Dérive des données (data drift)** : la distribution des entrées change (nouveaux
  commerces, changement de devise, panne d'une source). Elle est visible **immédiatement**,
  sans labels, en comparant les entrées au profil d'entraînement.
- **Dérive de concept (concept drift)** : la relation entre les variables et la fraude
  change (les fraudeurs adaptent leurs méthodes). Les entrées peuvent rester identiques :
  elle n'est **confirmée qu'avec les labels** (contestations, enquêtes), souvent plusieurs
  semaines plus tard.

## Indicateurs et seuils d'alerte

| Indicateur | Disponible | Seuil d'alerte | Source |
|---|---|---|---|
| Disponibilité de l'API (`/health`) | immédiat | 1 échec | supervision |
| Latence P95 de `/predict` | immédiat | > 50 ms sur 5 min | journaux JSON (`latency_ms`) |
| Taux d'erreurs 4xx / 5xx | immédiat | 5xx > 0 ; 422 > 1 % | journaux JSON (`status`) |
| Taux de transactions signalées | immédiat | ratio à la validation > 2 ou < 0,5 | `flag_rate_status` |
| Dérive par variable (PSI), fenêtres de 24 h | immédiat | > 0,10 surveillance ; > 0,25 alerte | `drift_report` |
| Lignes incomplètes | immédiat | > 1 % | `batch_quality_report` |
| Variable critique imputée | immédiat | toute occurrence -> revue manuelle | `variable_critique_manquante` |
| Montant ou composante hors domaine | immédiat | hausse soudaine du taux | `transaction_alerts` |
| Rappel, précision, FPR, FNR | différé (labels) | rappel < 0,75 ; FPR > 0,5 % | `evaluate_model` sur les labels reçus |
| Rappel par tranche de montant | différé | écart significatif entre tranches | audit par segment |

Prometheus et Grafana conviendraient en production (métriques exportées, tableaux de bord,
alertes) ; pour ce prototype, les indicateurs sont calculés par `app/monitoring.py` et les
alertes sont simulées.

## Simulation (`scripts/simulate_monitoring.py`)

| Lot | Incident injecté | Ce que voit la surveillance sans labels | Rappel à l'arrivée des labels |
|---|---|---|---|
| 1 à 3 | aucun | aucune alerte | 0,68 à 0,88 (variabilité, ~16 fraudes par lot) |
| 4 | montants x 3 | dérive « alerte » sur Amount (PSI 0,41) | 0,70 : dérive sans impact (le modèle utilise peu Amount) |
| 5 | V17, V10, V14 absents sur 5 % des lignes | 4,9 % de lignes incomplètes, 464 revues manuelles | 0,64 |
| 6 | fraudes déguisées (dérive de concept) | seulement une baisse du taux signalé (x 0,43) | **0,00** |

Constat préalable : par tranches horaires de 8 h, le cycle jour/nuit déclenche de fausses
alertes de dérive (Time, V1). La surveillance doit comparer des fenêtres de 24 h, ou chaque
tranche à la même tranche horaire de référence, et exclure Time du calcul (anomalie A09).

## Réponse aux alertes

1. **Dérive des données** : vérifier la source (panne, changement de format) ; si la dérive
   est réelle et durable, réentraîner puis passer la non-régression.
2. **Variable critique manquante** : revue manuelle de la transaction ; incident ouvert
   auprès de la source de données.
3. **Taux signalé anormal** : hausse -> risque de surcharge des analystes ; baisse -> risque
   de fraudes manquées (dérive de concept possible) : accélérer le retour des labels.
4. **Dégradation confirmée par les labels** : retour à la version précédente, réentraînement
   sur données récentes, requalification complète (ce plan de test).
