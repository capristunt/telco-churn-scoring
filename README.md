# Telco Customer Churn - Scoring de résiliation

Projet de scoring de churn pour TelcoWave (opérateur télécom mobile + fibre).
Objectif : estimer la probabilité de résiliation de chaque client afin de
prioriser un programme de rétention ciblé sous contrainte de budget marketing.

## 1. Contexte métier
La direction « Customer Success » souhaite réduire le churn au prochain trimestre
via des actions ciblées (appels sortants, remises, changement d'offre). Le modèle
de scoring sert à prioriser les clients les plus à risque.

## 2. Données
- Source : dataset public Kaggle « Telco Customer Churn ».
- Granularité : un enregistrement par client (7 043 clients).
- Cible : `Churn` (Yes/No).
- Features : démographie, services souscrits, contrat, facturation
  (`tenure`, `MonthlyCharges`, `TotalCharges`).
- Qualité connue : `TotalCharges` lu comme texte, ~11 valeurs vides
  (clients à `tenure = 0`).

## 3. Installation
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Reproduire

Le projet se déroule en trois notebooks à exécuter dans l'ordre :

```powershell
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_baseline_model.ipynb
jupyter notebook notebooks/03_finetuned_model.ipynb
```

- **`01_EDA.ipynb`** : analyse exploratoire, qualité des données, split stratifié
  60/20/20 persisté dans `data/processed/`.
- **`02_baseline_model.ipynb`** : Pipeline scikit-learn complète, `DummyClassifier`
  et `LogisticRegression` baseline, sérialisation dans `models/baseline.joblib`
  et scoring du test dans `reports/scoring_baseline_test.csv`.
- **`03_finetuned_model.ipynb`** : itérations (L1, HGB tuné), calibration
  isotonique, optimisation du seuil par coût/bénéfice, interprétabilité.
  Produit `models/finetuned.joblib` et `reports/scoring_finetuned_test.csv`.

Les modules `src/` (`data_prep.py`, `metrics.py`, `config.py`, `paths.py`)
factorisent le code partagé entre notebooks.

## 5. Résultats

**Modèle retenu** : `LogisticRegression(penalty="l2", class_weight="balanced")`
calibrée isotoniquement (`CalibratedClassifierCV(cv=5)`).

**Performance sur le test (1 406 clients)** :

| Métrique       | Baseline | Finetuné |
|----------------|----------|----------|
| ROC-AUC        | 0.8324   | 0.8325   |
| PR-AUC         | 0.6213   | 0.6153   |
| Recall@10 %    | 0.2647   | 0.2674   |
| Precision@10 % | 0.7071   | 0.7143   |

**Décision de ciblage** : seuil optimal `s* = 0.141` sur probabilité calibrée,
déterminé par optimisation du gain `105·TP − 15·FP` sur valid. À ce seuil sur
le test : 793 clients contactés (56 % de la base), 343 churners captés sur 374
(recall 91.7 %), gain net **29 265 €**.

Stratégie alternative pour budget contraint : `top 30 %` capture 82 % du gain
optimal en contactant ~30 % de la base.

## 6. Limites et pistes

- *Statisme du modèle.* Le scoring repose sur un instantané ; il faut prévoir un
  re-entraînement périodique pour suivre la dérive (saisonnalité, lancements
  d'offres, changement de mix client).
- *Granularité.* Aucun signal comportemental d'usage (consommation, incidents
  techniques, contacts service client). Enrichir le dataset avec ces signaux
  est probablement le levier de gain le plus important.
- *Pas de mesure d'effet causale.* Le modèle prédit le churn ; il ne mesure pas
  l'efficacité de l'offre de rétention. Un A/B test sur les clients ciblés est
  nécessaire pour estimer l'effet réel de la campagne et ajuster `saved_value`.
- *Sensibilité aux paramètres économiques.* Le seuil dépend du ratio
  `saved_value / offer_cost = 8`. Une variation de ces paramètres en
  conditions réelles déplace le seuil optimal — l'équipe métier devrait
  monitorer les vraies valeurs après campagne.