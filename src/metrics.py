"""Métriques d'évaluation et fonctions de reporting pour le scoring de churn."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)


def recall_precision_at_k(y_true: pd.Series, y_proba: pd.Series, k: float = 0.10,) -> tuple[float, float]:
    """Recall et precision sur les top k% des scores les plus élevés.

    Métriques alignées sur la stratégie de ciblage top K% sous budget marketing.
    """
    n_top = int(len(y_true) * k)
    order = np.argsort(y_proba)[::-1]
    top_idx = order[:n_top]
    y_top = y_true.iloc[top_idx] if hasattr(y_true, "iloc") else y_true[top_idx]
    recall = y_top.sum() / y_true.sum()
    precision = y_top.sum() / n_top
    return recall, precision


def evaluate(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series, k: float = 0.10,verbose: bool = True,) -> pd.Series:
    """Évalue une Pipeline sur (X, y) et retourne un dict des métriques clés.

    Métriques calculées : ROC-AUC, PR-AUC, recall@k, precision@k.
    Si verbose, affiche aussi le classification_report et la matrice de confusion.
    """
    y_proba = pipe.predict_proba(X)[:, 1]
    y_pred = pipe.predict(X)

    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    recall_k, precision_k = recall_precision_at_k(y, y_proba, k=k)

    if verbose:
        print(f"--- {name} ---")
        print(f"ROC-AUC          : {roc_auc:.4f}")
        print(f"PR-AUC           : {pr_auc:.4f}")
        print(f"Recall@{int(k*100)}%       : {recall_k:.4f}")
        print(f"Precision@{int(k*100)}%    : {precision_k:.4f}")
        print()
        print(classification_report(y, y_pred, target_names=["No churn", "Churn"]))
        print("Matrice de confusion :")
        print(confusion_matrix(y, y_pred))
        print()

    return pd.Series(
        {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            f"recall_{int(k*100)}": recall_k,
            f"precision_{int(k*100)}": precision_k,
        },
        name=name,
    )

def expected_gain(
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float,
    offer_cost: float = 15.0,
    saved_value: float = 120.0,
) -> float:
    """Gain espéré d'une campagne de rétention au seuil donné.

    Le client est contacté si ``y_proba >= threshold``. Coût d'offre fixe par contact,
    valeur sauvée uniquement si le client allait churner. Formule :

        gain = saved_value × TP - offer_cost × (TP + FP)
             = (saved_value - offer_cost) × TP - offer_cost × FP
    """
    y_pred = (y_proba >= threshold).astype(int)
    y_true_arr = y_true.values if hasattr(y_true, "values") else y_true
    tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    return (saved_value - offer_cost) * tp - offer_cost * fp


def find_optimal_threshold(
    y_true: pd.Series,
    y_proba: pd.Series,
    offer_cost: float = 15.0,
    saved_value: float = 120.0,
    n_steps: int = 1001,
) -> pd.DataFrame:
    """Balaye n_steps seuils de 0 à 1 et calcule gain, n_contacts, TP, FP par seuil.

    Retourne un DataFrame trié par seuil croissant, exploitable pour tracer la courbe
    gain vs seuil et localiser l'argmax via ``df.loc[df["gain"].idxmax()]``.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    y_true_arr = y_true.values if hasattr(y_true, "values") else y_true

    records = []
    for s in thresholds:
        y_pred = (y_proba >= s).astype(int)
        tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
        fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
        gain = (saved_value - offer_cost) * tp - offer_cost * fp
        records.append({
            "threshold": s,
            "n_contacts": tp + fp,
            "tp": tp,
            "fp": fp,
            "gain": gain,
        })
    return pd.DataFrame(records)

def compute_vif(frame: pd.DataFrame) -> pd.Series:
    """Compute the Variance Inflation Factor for each column.

    VIF_j = 1 / (1 - R²_j), where R²_j is the coefficient of
    determination obtained by regressing column j on all the others.
    A VIF above 5-10 signals problematic multicollinearity.

    Args:
        frame: DataFrame of numeric features without missing values.

    Returns:
        Series of VIF values indexed by column name, sorted descending.
    """
    vif = {}
    cols = list(frame.columns)
    for col in cols:
        others = [c for c in cols if c != col]
        model = LinearRegression().fit(frame.loc[:, others], frame.loc[:, col])
        r2 = model.score(frame.loc[:, others], frame.loc[:, col])
        vif[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return pd.Series(vif).sort_values(ascending=False)
