"""Préparation des données et construction du préprocesseur pour le scoring de churn.

Ce module centralise toute la logique de chargement, nettoyage et encodage,
de manière à ce que les notebooks 02 (baseline) et 03 (finetuning) partagent
strictement la même préparation. Cela garantit qu'une comparaison de modèles
ne reflète que des différences d'estimateur, pas de preprocessing.
"""
import itertools
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PROCESSED, DATA_RAW, TARGET


# Colonnes de services à compter pour la feature dérivée nb_services
SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
NO_SERVICE_VALUES = {"No", "No phone service", "No internet service"}

# Colonnes utilisées comme features par le modèle
NUM_COLS = ["tenure", "MonthlyCharges", "nb_services"]
CAT_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUM_COLS + CAT_COLS

# Colones retirées avant modélisation, justifications dans l'EDA
REMOVED_COLS = ["gender", "TotalCharges"]

# Colonnes exclues de la modélisation, justifications dans l'EDA
# customerID : identifiant, gender et PhoneService : Cramér's V proche de 0
EXCLUDED_COLS = ["customerID", "PhoneService"]

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage pré-Pipeline et création des features dérivées.

    Opérations effectuées dans cet ordre :
        - Retrait des colonnes ``gender`` et ``TotalCharges`` (voir EDA).
        - Retrait des clients à ``tenure = 0``.
        - Calcul de la feature ``nb_services``.
        - Encodage binaire de la cible dans la colonne ``churn_bin``.

    Args:
        df: DataFrame brut tel que chargé depuis le CSV source.

    Returns:
        DataFrame nettoyé, prêt à être passé au ColumnTransformer.
    """
    df = df.copy()
    df = df.drop(columns=REMOVED_COLS)
    df = df[df["tenure"] > 0]
    df.loc[:, "nb_services"] = df[SERVICE_COLS].apply(
        lambda row: sum(v not in NO_SERVICE_VALUES for v in row), axis=1
    )
    df.loc[:, "churn_bin"] = (df[TARGET] == "Yes").astype(int)
    return df


def load_splits(split_names: list[str] = ["train", "valid", "test"]) -> list[pd.DataFrame]:

    """Recharge le brut, reconstitue les splits par jointure sur les IDs persistés,
    applique ``prepare`` aux trois ensembles, et vérifie l'absence de chevauchement.

    Source de vérité unique : le CSV brut. Les fichiers ``split_*.csv`` ne
    contiennent que des listes d'IDs, ce qui évite toute désync entre données
    et partition.

    Args:
        split_names: Noms des splits à charger, dans l'ordre souhaité en retour.

    Returns:
        Tuple de DataFrames nettoyés dans l'ordre de ``split_names``.

    Raises:
        AssertionError: Si un ``customerID`` apparaît dans deux ensembles à la fois.
    """

    df = pd.read_csv(DATA_RAW)

    ids = {
        name: set(pd.read_csv(DATA_PROCESSED / f"split_{name}.csv")["customerID"])
        for name in split_names
    }

    for a, b in itertools.combinations(split_names, 2):
        assert ids[a] & ids[b] == set(), f"Chevauchement {a}/{b}"

    return tuple(
        prepare(df[df["customerID"].isin(ids[name])])
        for name in split_names
    )


def build_preprocessor() -> ColumnTransformer:
    """Construit le ColumnTransformer partagé entre baseline et finetuné.

    Deux branches :
        - Numériques : imputation médiane (robustesse) puis StandardScaler.
        - Catégorielles : OneHotEncoder avec ``drop="if_binary"`` pour traiter
          uniformément binaires et multiclasses, et ``handle_unknown="ignore"``
          pour ne pas casser sur une modalité absente du train.

    Returns:
        Un ``ColumnTransformer`` non-fitté, à intégrer dans une ``Pipeline``
        avec un estimateur en aval.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("encoder", OneHotEncoder(
            drop="if_binary", handle_unknown="ignore", sparse_output=False
        )),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUM_COLS),
        ("cat", categorical_pipe, CAT_COLS),
    ])