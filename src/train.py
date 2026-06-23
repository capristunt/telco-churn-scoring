"""Entraîne le modèle de churn final et sérialise l'artefact pour le serving."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.data_prep import FEATURES, build_preprocessor, load_splits


def build_pipeline() -> Pipeline:
    """Construit la pipeline complète : préprocesseur + LogReg L2 balanced calibrée.

    Returns:
        Pipeline scikit-learn prête à être fit.
    """
    base_clf = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="isotonic",
        cv=5,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", calibrated_clf),
        ]
    )


def main(output_path: Path) -> None:
    """Entraîne et sérialise le modèle final.

    Args:
        output_path: Chemin de destination de l'artefact .joblib.
    """
    # Train + valid concaténés pour maximiser les données vues par le modèle livré
    train_df, valid_df, _ = load_splits()
    fit_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    X_fit = fit_df[FEATURES]
    y_fit = fit_df["churn_bin"]
    print(f"Entraînement sur {len(X_fit)} clients (train + valid).")

    pipeline = build_pipeline()
    pipeline.fit(X_fit, y_fit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Modèle sauvegardé : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne le modèle de churn final.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/finetuned.joblib"),
        help="Chemin de sortie de l'artefact .joblib.",
    )
    args = parser.parse_args()
    main(args.output_path)