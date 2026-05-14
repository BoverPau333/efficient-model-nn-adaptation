"""Configuracion compartida para los experimentos"""

from pathlib import Path

import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    pass


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "datasets" / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

SEED = 42
EPOCHS = 5
LR = 1e-3
BATCH_SIZE = 32
NUM_WORKERS = 2

RETENTION_FRACTIONS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
