# Données

## Jeu complet (non versionné)

**Credit Card Fraud Detection** — Machine Learning Group, Université Libre de Bruxelles,
en collaboration avec Worldline. Distribué sur Kaggle :
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.

1. Télécharger l'archive depuis Kaggle (compte requis) : `archive.zip`.
2. Extraire `creditcard.csv` dans ce dossier : `data/creditcard.csv`.

Le fichier (~144 Mo) est exclu de Git par `.gitignore`. Empreinte du fichier utilisé :

| Caractéristique | Valeur |
|---|---|
| Fichier | `creditcard.csv` (150 828 752 octets) |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Transactions | 284 807, dont 492 fraudes (0,173 %) |
| Doublons exacts | 1 081 lignes, dont 19 fraudes (supprimés avant le découpage) |
| Valeurs manquantes | aucune |

Colonnes : `Time` (secondes écoulées depuis la première transaction du jeu),
`V1` à `V28` (composantes principales anonymisées : aucune interprétation métier),
`Amount` (montant en euros), `Class` (1 = fraude, 0 = normale).

## Échantillon versionné (`sample/creditcard_sample.csv`)

4 000 transactions tirées du jeu complet dédoublonné (graine 42) : **120 fraudes et
3 880 transactions normales**. L'échantillon est volontairement **enrichi en fraudes**
(3 % au lieu de 0,17 %) afin que les tests rapides de la CI disposent de cas positifs.
Ses métriques ne sont **pas représentatives** des performances réelles : il sert
uniquement à vérifier le comportement du code et la non-régression en CI.

Régénération : `python scripts/make_sample.py` (nécessite le jeu complet).

## Références

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson et Gianluca Bontempi.
*Calibrating Probability with Undersampling for Unbalanced Classification.*
IEEE Symposium Series on Computational Intelligence and Data Mining (CIDM), 2015.
