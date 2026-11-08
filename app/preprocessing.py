"""Contrat de données et préparation des variables.

Deux familles de traitements sont volontairement séparées :

* les traitements **sans état** (contrôle du schéma, conversion des types,
  remplacement des infinis) : ``validate_data``, ``clean_features`` et
  ``preprocess_data``. Ils n'apprennent rien des données ;
* les transformations **apprises** (imputation, passage au logarithme et
  standardisation) : regroupées dans ``build_preprocessor`` et ajustées uniquement
  à l'intérieur du ``Pipeline`` scikit-learn, sur le jeu d'entraînement.

Politique de traitement des entrées (voir docs/PLAN_DE_TEST.md) :

==============================  =======================================
Situation                       Comportement
==============================  =======================================
Objet non DataFrame, jeu vide   rejet (``DataValidationError``)
Colonne absente ou dupliquée    rejet
Type non numérique / booléen    rejet
Cible absente, manquante, != 0/1  rejet
Montant ou temps négatif        rejet
Colonne hors contrat            ignorée (journalisée)
Valeur infinie                  remplacée par une valeur manquante
Valeur manquante                imputée par la médiane d'entraînement
==============================  =======================================
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app.config import FEATURES, PCA_FEATURES, TARGET

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


class DataValidationError(ValueError):
    """Entrée qui ne respecte pas le contrat de données : elle est rejetée."""


def _is_numeric(series: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)


def validate_data(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Contrôle le schéma et les types, puis renvoie une copie limitée au contrat.

    Les colonnes sont renvoyées dans l'ordre du contrat (``FEATURES`` puis, le cas
    échéant, ``TARGET``). Les valeurs manquantes ou infinies des variables ne sont
    pas rejetées ici : elles relèvent de la stratégie de traitement documentée.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Un DataFrame pandas est attendu, reçu : {type(df).__name__}.")
    if df.shape[0] == 0:
        raise DataValidationError("Le jeu de données est vide.")

    duplicated = sorted(set(df.columns[df.columns.duplicated()]))
    if duplicated:
        raise DataValidationError(f"Colonnes dupliquées : {duplicated}.")

    required = FEATURES + ([TARGET] if require_target else [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes obligatoires absentes : {missing}.")

    extra = [col for col in df.columns if col not in FEATURES and col != TARGET]
    if extra:
        logger.warning("Colonnes hors contrat ignorées : %s", extra)

    wrong_types = [col for col in FEATURES if not _is_numeric(df[col])]
    if wrong_types:
        raise DataValidationError(f"Type incorrect (numérique attendu) : {wrong_types}.")

    for col in ("Amount", "Time"):
        values = df[col].to_numpy(dtype="float64")
        finite = np.isfinite(values)
        if (values[finite] < 0).any():
            raise DataValidationError(f"Valeurs négatives interdites dans « {col} ».")

    if require_target:
        target = df[TARGET]
        if target.isna().any():
            raise DataValidationError("La cible contient des valeurs manquantes.")
        if not _is_numeric(target):
            raise DataValidationError("La cible doit être numérique (0 ou 1).")
        invalid = sorted(set(pd.unique(target)) - {0, 1})
        if invalid:
            raise DataValidationError(f"Valeurs de cible invalides : {invalid} (0 ou 1 attendu).")

    return df.loc[:, required].copy()


def clean_features(X) -> pd.DataFrame:
    """Étape sans état : ordre des colonnes, conversion en float64, infinis -> NaN.

    Utilisée par ``preprocess_data`` et comme première étape du pipeline, afin que
    le modèle sauvegardé applique exactement le même nettoyage en production.
    """
    if isinstance(X, pd.DataFrame):
        missing = [col for col in FEATURES if col not in X.columns]
        if missing:
            raise DataValidationError(f"Colonnes obligatoires absentes : {missing}.")
        frame = X.loc[:, FEATURES]
    else:
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[1] != len(FEATURES):
            raise DataValidationError(
                f"Matrice de forme (n, {len(FEATURES)}) attendue, reçue : {array.shape}."
            )
        frame = pd.DataFrame(array, columns=FEATURES)
    try:
        frame = frame.astype("float64")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Conversion numérique impossible : {exc}") from exc
    return frame.replace([np.inf, -np.inf], np.nan)


def preprocess_data(
    df: pd.DataFrame, require_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Prépare les variables sans rien apprendre des données.

    Renvoie ``(X, y)`` : ``X`` contient les 30 variables du contrat en float64
    (infinis remplacés par NaN), ``y`` la cible en entier, ou ``None`` si
    ``require_target`` vaut ``False``.
    """
    data = validate_data(df, require_target=require_target)
    X = clean_features(data[FEATURES])
    y = data[TARGET].astype("int64") if require_target else None
    return X, y


# --- Transformations apprises (ajustées dans le Pipeline, sur l'entraînement) ------


def log_amount(values):
    """log(1 + montant) : réduit l'asymétrie des montants (borné à 0 par sécurité)."""
    return np.log1p(np.clip(np.asarray(values, dtype="float64"), 0, None))


def time_to_cyclic(values):
    """Encode l'heure de la journée (sin, cos) à partir des secondes écoulées."""
    hours = (np.asarray(values, dtype="float64") % SECONDS_PER_DAY) / 3600.0
    angle = 2 * np.pi * hours / 24.0
    return np.column_stack([np.sin(angle).ravel(), np.cos(angle).ravel()])


def cyclic_feature_names(_transformer, _input_features):
    return np.array(["Time_sin", "Time_cos"], dtype=object)


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def build_preprocessor(time_encoding: str = "raw") -> Pipeline:
    """Transformations apprises : imputation médiane, log du montant, standardisation.

    ``time_encoding`` : ``"raw"`` (temps standardisé) ou ``"cyclic"`` (heure de la
    journée encodée en sinus et cosinus, indépendante de la date de début du jeu).
    """
    amount = Pipeline(
        [
            ("impute", _imputer()),
            ("log", FunctionTransformer(log_amount, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    if time_encoding == "raw":
        time = Pipeline([("impute", _imputer()), ("scale", StandardScaler())])
    elif time_encoding == "cyclic":
        time = Pipeline(
            [
                ("impute", _imputer()),
                (
                    "cyclic",
                    FunctionTransformer(time_to_cyclic, feature_names_out=cyclic_feature_names),
                ),
            ]
        )
    else:
        raise ValueError(f"Encodage du temps inconnu : {time_encoding!r} (raw ou cyclic).")
    pca = Pipeline([("impute", _imputer()), ("scale", StandardScaler())])

    columns = ColumnTransformer(
        [("time", time, ["Time"]), ("pca", pca, PCA_FEATURES), ("amount", amount, ["Amount"])],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_features, feature_names_out="one-to-one")),
            ("columns", columns),
        ]
    )


# --- Outils sur le jeu de données ---------------------------------------------------


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Supprime les doublons exacts (risque de fuite entre entraînement et test)."""
    duplicated = df.duplicated(keep="first")
    features_dup = df.duplicated(subset=FEATURES, keep=False)
    conflicting = 0
    if TARGET in df.columns and features_dup.any():
        conflicting = int(
            df[features_dup].groupby(FEATURES, dropna=False)[TARGET].nunique().gt(1).sum()
        )
    report = {
        "rows_before": int(len(df)),
        "duplicates_removed": int(duplicated.sum()),
        "frauds_removed": int(df.loc[duplicated, TARGET].sum()) if TARGET in df else 0,
        "conflicting_labels": conflicting,
    }
    return df.loc[~duplicated].copy(), report


def canonical_order(df: pd.DataFrame) -> pd.DataFrame:
    """Trie les lignes de façon déterministe : le découpage ne dépend plus de l'ordre
    du fichier source (Kaggle, miroir, export retrié...)."""
    columns = [col for col in [*FEATURES, TARGET] if col in df.columns]
    return df.sort_values(columns, kind="mergesort").reset_index(drop=True)


def data_fingerprint(df: pd.DataFrame) -> str:
    """Empreinte SHA-256 du contenu (indépendante du format et de l'ordre du fichier)."""
    ordered = canonical_order(df)
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()
