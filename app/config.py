"""Paramètres centraux : contrat de données, graines, règles de décision et critères.

Les critères d'acceptation et la règle de choix du seuil sont fixés ici à partir
des exigences (plan de test, docs/PLAN_DE_TEST.md), AVANT toute évaluation sur le
jeu de test. Ils ne doivent pas être ajustés après observation des résultats de test.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FULL_DATA_PATH = DATA_DIR / "creditcard.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "creditcard_sample.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"
BASELINE_PATH = REPORTS_DIR / "baseline.json"
SAMPLE_BASELINE_PATH = ROOT / "tests" / "reference" / "baseline_sample.json"

# --- Contrat de données -------------------------------------------------------
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", *PCA_FEATURES, "Amount"]
TARGET = "Class"

# --- Reproductibilité ----------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20  # fraction du jeu complet réservée à l'évaluation finale
VALIDATION_SIZE = 0.20  # fraction du jeu complet utilisée pour le réglage

# --- Règle de choix du seuil (appliquée sur le jeu de VALIDATION) -----------------
# Hypothèses métier documentées dans le plan de test :
# - une fraude non détectée (FN) coûte en moyenne ~122 EUR (montant moyen d'une
#   fraude dans le jeu) + frais de gestion ;
# - une alerte injustifiée (FP) coûte ~5 EUR (revue par un analyste, gêne client).
# Rapport de coût retenu : 25 pour 1.
COST_FN = 25.0
COST_FP = 1.0
# Capacité de traitement des alertes : au plus 0,5 % des transactions normales.
MAX_FPR = 0.005

# --- Critères d'acceptation (vérifiés une seule fois sur le jeu de TEST) -------------
ACCEPTANCE_CRITERIA = {
    "recall": {"min": 0.75},  # au moins 3 fraudes sur 4 détectées
    "precision": {"min": 0.20},  # au moins 1 alerte sur 5 est une vraie fraude
    "fpr": {"max": 0.005},  # capacité de l'équipe de revue
    "pr_auc": {"min": 0.60},  # qualité de classement sur la classe rare
}

# --- Non-régression ---------------------------------------------------------------
# Dégradation maximale tolérée par rapport à la référence, sur le jeu de validation
# fixe. Au-delà de la moitié de la tolérance : investigation ; au-delà : rejet.
REGRESSION_TOLERANCES = {
    "pr_auc": 0.02,
    "recall": 0.03,  # environ 3 fraudes sur ~95 dans le jeu de validation
    "precision": 0.05,
    "fpr": 0.001,  # hausse absolue maximale du taux de faux positifs
}
HIGHER_IS_BETTER = {"pr_auc": True, "recall": True, "precision": True, "fpr": False}

# --- Surveillance -------------------------------------------------------------------
PSI_WARNING = 0.10
PSI_ALERT = 0.25
MISSING_RATE_ALERT = 0.01  # au-delà de 1 % de lignes incomplètes : alerte

# --- Performance (protocole : docs/PLAN_DE_TEST.md, section performance) -------------
# Volume observé dans le jeu : 284 807 transactions en 48 h, soit ~1,65 transaction/s.
# Exigence de dimensionnement : 100 fois le débit moyen observé.
PERFORMANCE_CRITERIA = {
    "single_p95_ms": 10.0,  # budget de scoring dans une autorisation de paiement
    "api_p95_ms": 50.0,
    "batch_throughput_min": 165.0,  # prédictions par seconde
    "model_size_max_mb": 10.0,
    "memory_max_mb": 500.0,
}
