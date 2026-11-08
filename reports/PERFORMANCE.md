# Rapport de performance

## Protocole

- Machine : Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 16 cœurs physiques / 22 logiques, 63.4 Go de RAM, Windows 11 (10.0.26300).
- Logiciels : Python 3.12.10, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1.
- Chronométrage `time.perf_counter`, 50 appels de chauffe avant chaque série.
- Prédiction isolée : 2 000 répétitions ; lots : 20 répétitions (5 pour le lot complet).
- Mémoire : processus séparé, RSS mesurée par psutil.
- API : uvicorn sur 127.0.0.1 (1 puis 4 processus), client httpx à connexion persistante, 2 000 requêtes séquentielles puis 16 clients concurrents x 250 requêtes. Client et serveur partagent la même machine (le client consomme aussi du CPU).

## Prédiction isolée

| Mesure | Moyenne (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|---|
| Modèle seul (predict_proba) | 3.753 | 3.659 | 4.383 | 5.584 |
| Modèle + validation du contrat | 4.848 | 4.723 | 5.618 | 7.178 |
| Requête API /predict (HTTP + validation JSON) | 7.759 | 7.650 | 8.660 | 10.752 |
| Requête API, 16 clients concurrents, 1 processus | 120.449 | 118.655 | 150.822 | 169.276 |
| Requête API, 16 clients concurrents, 4 processus | 33.988 | 35.947 | 47.793 | 56.921 |
| Requête API /predict/batch (100 transactions) | 10.940 | 10.835 | 11.737 | 12.288 |
| Charge imposée 165 req/s, 1 processus | 2673.393 | 2739.722 | 4860.422 | 5018.300 |
| Charge imposée 165 req/s, 4 processus | 27.540 | 10.266 | 18.104 | 56.892 |

## Débit par taille de lot

| Taille du lot | P50 (ms) | P95 (ms) | Débit (prédictions/s) |
|---|---|---|---|
| 1 | 4.840 | 6.202 | 207 |
| 10 | 4.563 | 4.878 | 2191 |
| 100 | 4.513 | 5.013 | 22157 |
| 1000 | 4.968 | 5.429 | 201282 |
| 10000 | 9.881 | 10.382 | 1012023 |
| 56746 | 34.332 | 34.887 | 1652836 |

API, 16 clients concurrents : 125 requêtes/s avec 1 processus (erreurs 0.00%), 338 requêtes/s avec 4 processus (erreurs 0.00%).

## Ressources

| Mesure | Valeur |
|---|---|
| Taille du modèle (octets) | 18654 |
| Temps de chargement du modèle (ms) | 3.030 |
| python_mb | 17.855 |
| after_imports_mb | 145.137 |
| after_model_load_mb | 146.473 |
| after_full_test_scoring_mb | 163.137 |
| RSS du serveur API, 1 processus (Mo) | 180.281 |
| RSS du serveur API, 4 processus + superviseur (Mo) | 712.828 |

## Critères d'acceptation

Volume du jeu : 284 807 transactions en 48 h, soit 1,65 transaction/s en moyenne. Exigence de dimensionnement retenue : 100 fois ce débit (165/s). Budget de latence d'une autorisation de paiement : quelques centaines de millisecondes, dont 10 ms allouées au modèle et 50 ms à l'appel du service de scoring.

| Contrôle | Mesure | Exigence | Résultat |
|---|---|---|---|
| Latence P95 d'une prédiction (modèle + validation) | 5.62 ms | <= 10.0 ms | conforme |
| Latence P95 d'une requête API /predict | 8.66 ms | <= 50.0 ms | conforme |
| Débit max API, 1 processus (saturation) | 124.73 req/s | >= 165.0 req/s | NON CONFORME |
| Débit max API, 4 processus (saturation) | 337.92 req/s | >= 165.0 req/s | conforme |
| P95 API à 165 req/s imposées, 1 processus | 4860.42 ms | <= 50.0 ms | NON CONFORME |
| P95 API à 165 req/s imposées, 4 processus | 18.10 ms | <= 50.0 ms | conforme |
| Débit par lot | 1652836.23 préd./s | >= 165.0 préd./s | conforme |
| Taille du modèle | 0.02 Mo | <= 10.0 Mo | conforme |
| Mémoire du service, 1 processus (RSS) | 180.28 Mo | <= 500.0 Mo | conforme |
| Mémoire du service, 4 processus (RSS) | 712.83 Mo | <= 2000.0 Mo | conforme |
