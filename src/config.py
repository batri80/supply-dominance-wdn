"""Paths and settings shared by every script.

Importing this module also changes the working directory to the outputs folder,
so every script reads and writes its results there, wherever it is started from.

Override any location with environment variables:
  PSEP_DATA     folder with L-TOWN.inp, scada/ and BattLeDIM/   (default: <package>/data)
  PSEP_OUT      folder for all results                            (default: <package>/outputs)
  PSEP_WORKERS  parallel simulation processes                     (default: CPU count - 1)
"""
import os
import warnings

warnings.filterwarnings("ignore")          # WNTR and pandas emit many harmless warnings

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DATA = os.path.abspath(os.environ.get("PSEP_DATA", os.path.join(ROOT, "data")))
OUT = os.path.abspath(os.environ.get("PSEP_OUT", os.path.join(ROOT, "outputs")))
WORKERS = int(os.environ.get("PSEP_WORKERS", max(1, (os.cpu_count() or 2) - 1)))

LTOWN_INP = os.path.join(DATA, "L-TOWN.inp")
SCADA_DIR = os.path.join(DATA, "scada")                                  # 2018_SCADA.xlsx, 2019_SCADA.xlsx
BATTLEDIM_SCORING = os.path.join(DATA, "BattLeDIM", "Scoring Algorithm")  # from the KIOS repository

os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)
