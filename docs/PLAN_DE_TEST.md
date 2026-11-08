# Plan de test - AI-TestGuard

Système qualifié : détection de fraude sur transactions par carte (jeu ULB / Kaggle),
pipeline scikit-learn et API de scoring. Auteur : Joe MAGNAN - MDSIA 2.

## 1. Périmètre

| Composant | Dans le périmètre | Hors périmètre |
|---|---|---|
| Données | contrat de schéma et de types, cible, doublons, valeurs manquantes ou infinies, échantillon CI | collecte, droits d'usage des données réelles d'une banque |
| Prétraitement | nettoyage sans état, imputation, log du montant, standardisation (dans le `Pipeline`) | ingénierie de variables métier |
| Modèle | régression logistique et arbre de décision, choix des hyperparamètres et du seuil | modèles lourds (forêts, boosting, réseaux) |
| Prédiction | probabilité, décision au seuil, alertes de qualité | blocage automatique d'une carte |
| Intégration | sauvegarde, rechargement contrôlé, API FastAPI, CI GitHub Actions | infrastructure de production, authentification, chiffrement |

## 2. Exigences

**Fonctionnelles** : rejeter toute entrée hors contrat ; traiter selon une règle documentée
les valeurs manquantes ou infinies ; produire une probabilité valide et une décision au seuil ;
détecter au moins 75 % des fraudes en limitant les alertes injustifiées ; signaler les
entrées atypiques et la dérive.

**Non fonctionnelles** : reproductibilité (graines, versions, empreintes), absence de fuite
entre entraînement et test, non-régression, latence et débit compatibles avec une
autorisation de paiement, journaux sans données sensibles, aucun secret dans le dépôt,
exécution automatique des tests.

## 3. Hypothèses métier et critères d'acceptation

Fixés **avant** l'évaluation sur le jeu de test (commit `6064d6f`, fichier `app/config.py`).

| Élément | Valeur | Justification |
|---|---|---|
| Coût d'une fraude manquée (FN) | 25 | montant moyen d'une fraude dans le jeu : 122 EUR, plus frais de gestion |
| Coût d'une alerte injustifiée (FP) | 1 | environ 5 EUR : revue par un analyste et gêne du client |
| Contrainte de capacité | FPR <= 0,5 % | volume d'alertes absorbable par l'équipe de revue |
| Rappel minimal (test) | >= 0,75 | au moins 3 fraudes sur 4 détectées |
| Précision minimale (test) | >= 0,20 | au moins 1 alerte sur 5 fondée |
| FPR maximal (test) | <= 0,5 % | capacité de revue |
| PR-AUC minimale (test) | >= 0,60 | qualité de classement sur la classe rare |
| Non-régression | PR-AUC -0,02 ; rappel -0,03 ; précision -0,05 ; FPR +0,001 | au-delà : rejet ; au-delà de la moitié : investigation |
| Performance | P95 modèle <= 10 ms ; P95 API <= 50 ms ; débit >= 165/s ; modèle <= 10 Mo ; mémoire <= 500 Mo | 1,65 transaction/s en moyenne dans le jeu, dimensionnement à 100 fois |

**Règles de décision.** Le modèle est **rejeté** si un test critique (T01 à T06, T08, T09,
T13, T15) échoue, si une fuite est détectée, ou si un critère d'acceptation n'est pas atteint
sur le test. Il est **qualifié techniquement pour un pilote** si tous les critères sont atteints ;
les anomalies majeures ouvertes conditionnent alors le pilote. La **mise en production** exige
en plus une validation métier, opérationnelle, sécurité et réglementaire.

## 4. Matrice de traçabilité

| ID | Exigence | Type | Risque maîtrisé | Cas de test (fichier) | Résultat attendu | Statut |
|---|---|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Données | prédiction calculée sur une entrée incomplète | `test_missing_required_column_is_rejected` (x4), `test_valid_data_keeps_all_contract_columns_in_order` (test_preprocessing) | aucune colonne indispensable manquante ; sinon rejet explicite | réussi |
| T02 | Les données respectent les types attendus | Unitaire | conversion implicite (texte, booléen -> 1.0) | `test_wrong_type_is_rejected` (x3), `test_boolean_column_is_rejected`, `test_invalid_transactions_are_rejected` (API) | types conformes au contrat ; sinon rejet | réussi |
| T03 | Le prétraitement ne produit pas de valeurs non finies | Unitaire | calcul NaN/infini propagé au modèle | `test_preprocessor_output_is_finite_despite_nan_and_inf`, `test_property_finite_inputs_give_finite_outputs` (40 cas Hypothesis) | aucune valeur NaN ou infinie en sortie | réussi |
| T04 | Le modèle produit une probabilité valide | Fonctionnel ML | sortie inexploitable | `test_prediction_probabilities_are_valid`, `test_property_any_valid_transaction_gets_valid_probability` (60 cas), `test_output_contract_violation_is_detected` | probabilité finie entre 0 et 1 | réussi |
| T05 | Les performances sont évaluées sur des données indépendantes | ML | performance surestimée | `test_partitions_do_not_overlap`, `test_scaler_is_fitted_on_training_data_only`, `test_test_metrics_are_reported_but_threshold_comes_from_validation` | jeu de test non utilisé pour le réglage | réussi |
| T06 | Les résultats sont reproductibles | Non-régression | résultat impossible à rejouer ou à auditer | `test_retraining_with_same_seed_is_reproducible`, `test_reloaded_pipeline_gives_identical_predictions` (tolérance 1e-12), `test_full_retraining_reproduces_baseline` (jeu complet) | écart nul à 1e-12 près | réussi |
| T07 | Les tests s'exécutent sur GitHub | CI | régression livrée sans être vue | `.github/workflows/tests.yml` | workflow réussi, échec volontaire correctement signalé | réussi |
| T08 | La cible ne contient que 0 et 1, sans valeur manquante | Données | étiquettes corrompues qui faussent apprentissage et métriques | `test_target_outside_zero_one_is_rejected` (x3), `test_missing_target_is_rejected` | rejet explicite | réussi |
| T09 | Aucune transaction n'apparaît à la fois en entraînement et en test | Données / fuite | 1 081 doublons exacts dans le jeu : performance surestimée | `test_deduplicate_removes_exact_duplicates`, `test_duplicates_are_removed_before_split`, `test_partitions_do_not_overlap` | doublons supprimés avant le découpage | réussi |
| T10 | Le seuil est choisi sur la validation selon le coût FN/FP et la capacité | ML | seuil arbitraire (0,5) ou réglé sur le test | `test_select_threshold_*` (x4), `test_threshold_matches_baseline` | coût minimal sous FPR <= 0,5 % | réussi |
| T11 | Les critères d'acceptation métier sont atteints sur le test | ML / métier | fraudes manquées, surcharge des analystes | `reports/metrics.json` (`acceptance`), `test_acceptance_criteria` | rappel, précision, FPR et PR-AUC conformes | atteints (rappel limite, voir A05) |
| T12 | Une nouvelle version ne dégrade pas la référence au-delà des tolérances | Non-régression | dégradation silencieuse | `test_current_code_does_not_regress_against_frozen_baseline`, `test_training_run_never_modifies_the_baseline`, `scripts/compare_versions.py` | décision accepter / investiguer / rejeter motivée | réussi |
| T13 | Le modèle chargé est intègre | Sécurité | fichier altéré ou corrompu, code exécuté au chargement | `test_tampered_model_is_refused`, `test_corrupted_file_with_matching_hash_is_reported`, `test_unexpected_bundle_format_is_refused` (x3) | refus explicite (`ModelLoadingError`) | réussi |
| T14 | Chaque scénario de robustesse a un comportement défini (rejeter, traiter, alerter) | Robustesse | comportement silencieux sur entrée dégradée | `tests/test_robustness.py` (19 tests), `scripts/run_robustness.py` | comportement conforme au tableau de la partie V | réussi, anomalie A02 |
| T15 | L'API valide les entrées et les fichiers | Sécurité opérationnelle | entrée malformée acceptée, erreur 500 | `test_invalid_transactions_are_rejected` (x6), `test_missing_field_is_rejected`, `test_non_finite_values_are_rejected`, `test_invalid_files_are_rejected` (x5), lot > 1 000 | rejet 413 / 415 / 422, jamais 500 | réussi après correction A01 |
| T16 | Les journaux et les réponses d'erreur ne contiennent aucune donnée de transaction | Sécurité | fuite de données sensibles | `test_logs_contain_no_transaction_values`, `test_error_responses_do_not_echo_submitted_values` | aucune valeur saisie dans les journaux ni dans les erreurs | réussi |
| T17 | Aucun secret ni jeu complet dans Git ; dépendances figées | Sécurité | fuite de secrets, données non autorisées, build non reproductible | `tests/test_security.py` (6 tests) | aucun motif détecté | réussi |
| T18 | Latence, débit, taille et mémoire conformes | Performance | autorisation de paiement trop lente, service saturé | `tests/test_performance.py`, `scripts/run_performance.py` | critères de la section 3 | conforme avec 4 processus (A06) |
| T19 | La dérive et les incidents de données sont détectés sans labels | Surveillance | dégradation non vue en production | `tests/test_monitoring.py` (13 tests), `scripts/simulate_monitoring.py` | alerte PSI, données manquantes, variables critiques, taux signalé | réussi, limite A11 |
| T20 | Le code respecte le style et la couverture minimale | Qualité | dette, code mort non testé | Ruff (CI), `--cov-fail-under=90` | 0 erreur Ruff, couverture >= 90 % | réussi (98 %) |

## 5. Outils et preuves conservées

| Outil | Usage | Preuve conservée |
|---|---|---|
| pytest, pytest-cov | 179 tests, couverture | `reports/tests/junit.xml`, `coverage.xml`, artefacts CI |
| Hypothesis | génération de cas (propriétés) | tests de propriété dans `test_preprocessing.py`, `test_robustness.py` |
| scikit-learn, joblib | pipeline, sauvegarde | `models/fraud_model.joblib`, `models/metadata.json` (SHA-256, versions, graines) |
| FastAPI, httpx, uvicorn | API et tests de charge | `reports/PERFORMANCE.md`, `reports/performance.json` |
| time.perf_counter, psutil, tracemalloc | performance et mémoire | `reports/performance.json` |
| Ruff | style | journaux CI |
| GitHub Actions | automatisation, protection de `main` | historique des exécutions, PR d'échec volontaire |
| Git | traçabilité | commit `6064d6f` (critères fixés avant évaluation), référence `reports/baseline.json` |

## 6. Organisation de l'exécution

| Moment | Tests | Durée |
|---|---|---|
| Chaque push et pull request vers `main` | Ruff + 178 tests rapides (échantillon versionné) + couverture | environ 1 min |
| Manuel ou chaque lundi | + ré-entraînement complet et comparaison à la référence | quelques minutes |
| Avant une nouvelle version | `scripts/compare_versions.py`, `run_robustness.py`, `run_performance.py`, `simulate_monitoring.py` (jeu complet) | quelques minutes |
