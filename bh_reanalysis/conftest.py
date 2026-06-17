import sys
from pathlib import Path

# Make `src` importable as a package from anywhere inside bh_reanalysis/
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
