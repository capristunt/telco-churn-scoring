"""Préparation des données et construction du préprocesseur pour le scoring de churn.

Ce module centralise toute la logique de chargement, nettoyage et encodage,
de manière à ce que les notebooks 02 (baseline) et 03 (finetuning) partagent
strictement la même préparation. Cela garantit qu'une comparaison de modèles
ne reflète que des différences d'estimateur, pas de preprocessing.
"""
from pathlib import Path
from typing import Tuple

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
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "nb_services"]
CAT_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUM_COLS + CAT_COLS

# Colonnes exclues de la modélisation, justifications dans l'EDA
# customerID : identifiant, gender et PhoneService : Cramér's V proche de 0
EXCLUDED_COLS = ["customerID", "gender", "PhoneService"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage pré-Pipeline et création des features dérivées.

    Opérations effectuées dans cet ordre :
        1. Retrait de la colonne ``gender`` (non informative).
        2. Retrait des clients à ``tenure = 0``.
        3. Conversion de ``TotalCharges`` en numérique.
        4. Calcul de la feature ``nb_services``.
        5. Encodage binaire de la cible dans la colonne ``churn_bin``.

    Args:
        df: DataFrame brut tel que chargé depuis le CSV source.

    Returns:
        DataFrame nettoyé, prêt à être passé au ColumnTransformer.
    """
    df = df.copy()
    df = df.drop(columns=["gender"])
    df = df[df["tenure"] > 0]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["nb_services"] = df[SERVICE_COLS].apply(
        lambda row: sum(v not in NO_SERVICE_VALUES for v in row), axis=1
    )
    df["churn_bin"] = (df[TARGET] == "Yes").astype(int)
    return df


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recharge le brut, reconstitue les splits par jointure sur les IDs persistés,
    applique ``prepare`` aux trois ensembles, et vérifie l'absence de chevauchement.

    Source de vérité unique : le CSV brut. Les fichiers ``split_*.csv`` ne
    contiennent que des listes d'IDs, ce qui évite toute désync entre données
    et partition.

    Returns:
        Tuple ``(train_df, valid_df, test_df)`` contenant les trois ensembles
        nettoyés et enrichis des features dérivées.

    Raises:
        AssertionError: Si un ``customerID`` apparaît dans deux ensembles à la fois.
    """
    df = pd.read_csv(DATA_RAW)

    ids_train = pd.read_csv(DATA_PROCESSED / "split_train.csv")["customerID"]
    ids_valid = pd.read_csv(DATA_PROCESSED / "split_valid.csv")["customerID"]
    ids_test = pd.read_csv(DATA_PROCESSED / "split_test.csv")["customerID"]

    assert set(ids_train) & set(ids_valid) == set(), "Chevauchement train/valid"
    assert set(ids_train) & set(ids_test) == set(), "Chevauchement train/test"
    assert set(ids_valid) & set(ids_test) == set(), "Chevauchement valid/test"

    train_df = prepare(df[df["customerID"].isin(ids_train)])
    valid_df = prepare(df[df["customerID"].isin(ids_valid)])
    test_df = prepare(df[df["customerID"].isin(ids_test)])

    return train_df, valid_df, test_df


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