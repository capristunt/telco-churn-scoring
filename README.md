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