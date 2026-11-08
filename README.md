# AI-TestGuard

[![Tests AI-TestGuard](https://github.com/Joemagnan/AI-TestGuard/actions/workflows/tests.yml/badge.svg)](https://github.com/Joemagnan/AI-TestGuard/actions/workflows/tests.yml)

Validation, robustesse et fiabilité d'un modèle de détection de fraude bancaire
(jeu *Credit Card Fraud Detection*, ULB / Kaggle). Évaluation formative de test logiciel,
Joe MAGNAN - MDSIA 2.

L'objectif n'est pas d'obtenir le meilleur score, mais de **démontrer, par des tests
reproductibles, dans quelles conditions le modèle est acceptable**.

## Résultat de la qualification

| | |
|---|---|
| Modèle livré (v1.0.0) | arbre de décision (profondeur 8), seuil 0,148 choisi sur la validation |
| Test (56 746 transactions, 95 fraudes) | 72 fraudes détectées, 23 manquées, 33 fausses alertes |
| Rappel / précision / FPR / PR-AUC | 0,758 / 0,686 / 0,058 % / 0,695 : **4 critères sur 4 atteints** |
| Tests automatisés | 179 (178 en CI à chaque PR), couverture 98 % |
| Anomalies | 14 enregistrées : 4 corrigées, 2 atténuées (dont la critique), 8 ouvertes ou documentées |
| Décision | **apte à un pilote encadré** (sans blocage automatique), **non autorisé en production** |

Détail : [rapport d'évaluation](reports/EVALUATION_ML.md),
[robustesse](reports/ROBUSTESSE.md), [non-régression](reports/NON_REGRESSION.md),
[performance](reports/PERFORMANCE.md), [anomalies](docs/ANOMALIES.md).

## Structure

```
AI-TestGuard/
├── app/
│   ├── preprocessing.py   contrat de données, nettoyage, transformations apprises
│   ├── training.py        découpage, entraînement, choix du seuil
│   ├── prediction.py      sauvegarde, chargement contrôlé (SHA-256), prédiction
│   ├── evaluation.py      métriques, intervalles de confiance, non-régression
│   ├── monitoring.py      profil de référence, PSI, alertes
│   ├── pipeline.py        pipeline de bout en bout (python -m app.pipeline)
│   ├── api.py             API FastAPI de scoring
│   ├── plots.py           graphiques
│   └── config.py          critères d'acceptation, tolérances, graines
├── tests/                 179 tests (dont les 5 fichiers demandés)
├── scripts/               campagnes sur le jeu complet, outils CI
├── data/                  README (source, empreinte) + échantillon CI versionné
├── models/                modèle livré + metadata.json
├── reports/               métriques, rapports, figures, référence figée
├── docs/                  plan de test, anomalies, audit des biais, surveillance
├── .github/workflows/tests.yml
├── requirements.txt
└── pyproject.toml         configuration Ruff, pytest, couverture
```

## Utilisation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows ; source .venv/bin/activate sous Linux
pip install -r requirements.txt
```

Tests rapides (sans le jeu complet, comme la CI) :

```bash
pytest -m "not slow" --cov=app
ruff check . && ruff format --check .
```

Avec le jeu complet (`data/creditcard.csv`, voir [data/README.md](data/README.md)) :

```bash
python -m app.pipeline --data data/creditcard.csv   # entraînement + évaluation finale
pytest -m slow                                      # ré-entraînement = référence ?
python scripts/run_robustness.py                    # PARTIE V
python scripts/compare_versions.py                  # PARTIE VI
python scripts/run_performance.py                   # PARTIE VIII
python scripts/run_bias_audit.py                    # PARTIE VII
python scripts/simulate_monitoring.py               # PARTIE X
```

API :

```bash
uvicorn app.api:app --port 8000 --workers 4
# http://127.0.0.1:8000/docs
```

## Intégration continue

`.github/workflows/tests.yml` exécute Ruff, puis les 178 tests rapides avec la couverture
(seuil 90 %), à chaque push et pull request vers `main`. La branche `main` est protégée :
une pull request ne peut être fusionnée que si les deux contrôles réussissent. Le
ré-entraînement complet (jeu récupéré depuis le miroir OpenML du jeu ULB) et la comparaison
à la référence sont lancés manuellement ou chaque lundi.

## Livrables du sujet

| Livrable | Emplacement |
|---|---|
| Dépôt GitHub (code, tests, dépendances, documentation) | ce dépôt |
| Plan de test (exigences, risques, cas, critères) | [docs/PLAN_DE_TEST.md](docs/PLAN_DE_TEST.md) |
| Suite de tests automatisés | [tests/](tests/) |
| Rapport d'évaluation ML | [reports/EVALUATION_ML.md](reports/EVALUATION_ML.md), [reports/figures/](reports/figures/) |
| Rapport de robustesse et de non-régression | [reports/ROBUSTESSE.md](reports/ROBUSTESSE.md), [reports/NON_REGRESSION.md](reports/NON_REGRESSION.md) |
| Rapport de performance | [reports/PERFORMANCE.md](reports/PERFORMANCE.md) |
| Note d'audit des biais | [docs/AUDIT_BIAIS.md](docs/AUDIT_BIAIS.md) |
| Stratégie de surveillance | [docs/SURVEILLANCE.md](docs/SURVEILLANCE.md) |
| Workflow GitHub Actions | [.github/workflows/tests.yml](.github/workflows/tests.yml) |
| Rapport final de qualification | document Word remis séparément |

## Données

Jeu ULB / Worldline distribué sur Kaggle, non versionné (voir [data/README.md](data/README.md)).
Les résultats portent sur des transactions européennes de septembre 2013 : ils ne prouvent
pas, à eux seuls, la performance sur des transactions actuelles.
