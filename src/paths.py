"""Rutas del proyecto, resueltas desde la ubicación de este archivo.

Así los notebooks funcionan igual se ejecuten desde la raíz del repo o desde
notebooks/ (antes eda usaba 'data/' y 02_modeling '../data/').
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
AUDIO = {"HS": DATA_DIR / "HS", "LS": DATA_DIR / "LS", "Mix": DATA_DIR / "Mix"}

OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def add_root_to_path():
    """Permite `from src... import ...` desde un notebook, sea cual sea el cwd."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return ROOT
