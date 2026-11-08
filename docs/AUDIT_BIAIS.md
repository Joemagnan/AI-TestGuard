# Note d'audit des biais

## Limite essentielle

Le jeu Credit Card Fraud ne contient **aucun attribut démographique** : ni âge, ni sexe, ni
localisation, ni catégorie socioprofessionnelle. Les variables V1 à V28 sont des composantes
principales anonymisées. Il est donc **impossible de conclure à l'équité** du système envers
des groupes de personnes. Conclusion à retenir : « équité non évaluable avec ces données »,
et non « système équitable ».

Aucun attribut n'a été déduit des composantes anonymisées : ce serait à la fois
méthodologiquement infondé et contraire au principe de minimisation des données.

## Ce qui a pu être mesuré : segments opérationnels

À défaut de groupes de personnes, les performances ont été comparées sur des segments
observables et légitimes (montant, heure), avec des intervalles de confiance à 95 %
(bootstrap stratifié, 1 000 tirages). Source : `reports/SEGMENTS.md`.

| Segment | Fraudes | Rappel [IC 95 %] | FPR [IC 95 %] | Précision [IC 95 %] |
|---|---|---|---|---|
| Montant < 10 EUR | 47 | 0,70 [0,57 ; 0,83] | 0,072 % [0,036 ; 0,114] | 0,70 [0,60 ; 0,83] |
| Montant 10-100 EUR | 22 | 1,00 [1,00 ; 1,00] | 0,031 % [0,012 ; 0,054] | 0,73 [0,61 ; 0,88] |
| Montant >= 100 EUR | 26 | 0,65 [0,46 ; 0,81] | 0,096 % [0,044 ; 0,166] | 0,61 [0,45 ; 0,76] |
| Nuit (0 h - 6 h) | 27 | 0,81 [0,63 ; 0,96] | 0,062 % [0,000 ; 0,123] | 0,88 [0,77 ; 1,00] |
| Jour (6 h - 24 h) | 68 | 0,74 [0,63 ; 0,84] | 0,058 % [0,039 ; 0,077] | 0,62 [0,54 ; 0,72] |

Lecture :

- les fraudes de **100 EUR et plus**, les plus coûteuses, sont les moins bien détectées, et
  l'écart avec la tranche 10-100 EUR est net (intervalles disjoints) ;
- les clients qui font des achats de 100 EUR et plus subissent environ **3 fois plus de
  fausses alertes** que ceux de la tranche 10-100 EUR ;
- l'écart jour / nuit n'est pas significatif (intervalles qui se recouvrent largement).

Les effectifs de fraudes par segment (22 à 68) restent faibles : ces écarts sont des signaux
à confirmer, pas des preuves.

## Données nécessaires pour un véritable audit d'équité

1. Des attributs de groupe **légalement utilisables** et collectés avec une base légale
   (par exemple tranche d'âge, région, type de carte ou de client), sous contrôle du délégué
   à la protection des données.
2. Des effectifs suffisants de fraudes par groupe (plusieurs centaines) pour des
   intervalles de confiance exploitables.
3. Des labels fiables et récents, y compris pour les transactions bloquées (sinon les
   fausses alertes ne sont jamais corrigées : biais de sélection).
4. Une définition, validée par le métier et la conformité, des écarts acceptables
   (parité des taux de faux positifs, égalité des rappels...).

Outils adaptés une fois ces données disponibles : Fairlearn (`MetricFrame`) ou AI Fairness 360.

## Recommandations

- Ne pas annoncer d'équité tant que l'audit ci-dessus n'est pas réalisé.
- Intégrer le montant au coût des faux négatifs (anomalie A12) et suivre le rappel par
  tranche de montant dans la surveillance.
- Réévaluer les segments à chaque nouvelle version (non-régression par segment).
