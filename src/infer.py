"""Score un CSV de clients avec le modèle final et sauvegarde les prédictions."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data_prep import FEATURES, prepare

# Seuil de décision optimal identifié dans le notebook 03 (gain 105*TP - 15*FP)
DEFAULT_THRESHOLD = 0.141


def assign_segment(proba: pd.Series) -> pd.Series:
    """Assigne un segment de risque par quartile de probabilité sur le batch.

    Args:
        proba: Série des probabilités de churn.

    Returns:
        Série catégorielle Q1 (bas) à Q4 (haut), Q4 = 25% les plus à risque.
    """
    return pd.qcut(
        proba,
        q=4,
        labels=["Q1 (bas)", "Q2", "Q3", "Q4 (haut)"],
    )


def main(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    threshold: float,
) -> None:
    """Score le CSV et sauvegarde les prédictions.

    Args:
        input_path: CSV brut au format TelcoWave.
        output_path: CSV de sortie (customerID, proba_churn, label_pred, risk_segment).
        model_path: Artefact .joblib à charger.
        threshold: Seuil de décision pour la binarisation.
    """
    df_raw = pd.read_csv(input_path)
    print(f"Chargé {len(df_raw)} clients depuis {input_path}")

    # prepare() retire les clients tenure=0 (out-of-scope EDA)
    df_prepared = prepare(df_raw)
    n_excluded = len(df_raw) - len(df_prepared)
    if n_excluded > 0:
        print(f"  {n_excluded} client(s) exclu(s) (tenure = 0).")

    pipeline = joblib.load(model_path)
    proba = pipeline.predict_proba(df_prepared[FEATURES])[:, 1]

    predictions = pd.DataFrame({
        "customerID": df_prepared["customerID"].values,
        "proba_churn": proba,
        "label_pred": (proba >= threshold).astype(int),
        "risk_segment": assign_segment(pd.Series(proba)),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    n_positive = int(predictions["label_pred"].sum())
    print(f"Prédictions sauvegardées : {output_path}")
    print(f"  {len(predictions)} clients scorés, {n_positive} prédits à risque (seuil {threshold}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score un CSV de clients avec le modèle final.")
    parser.add_argument("--input-path", type=Path, required=True, help="CSV brut d'entrée.")
    parser.add_argument("--output-path", type=Path, required=True, help="CSV de sortie des prédictions.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/finetuned.joblib"),
        help="Artefact .joblib à charger (défaut: models/finetuned.joblib).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Seuil de décision (défaut: {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.model_path, args.threshold)