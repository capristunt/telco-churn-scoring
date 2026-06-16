"""Constantes partagées du projet"""
from pathlib import Path

# Roots
ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_RAW = ROOT / "data" / "raw" / "telco_customer_churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproductibilité
RANDOM_STATE = 42

# Target
TARGET = "Churn"