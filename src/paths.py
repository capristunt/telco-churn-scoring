"""Utilitaires de manipulation de chemins."""
from pathlib import Path
from src.config import ROOT

def rel(path: Path) -> str:
    """Retourne le chemin relatif à la racine du projet, sous forme de chaîne.

    Utilisé pour raccourcir l'affichage des chemins dans les logs et sorties de
    notebooks, en supprimant le préfixe absolu de la machine.

    Returns:
        Chaîne du chemin relatif au dossier parent de ROOT, par exemple
        "telco-churn-scoring/models/baseline.joblib".
    """
    return str(Path(path).relative_to(ROOT.parent))