"""Держит все пути MNE / MOABB / matplotlib внутри projects/bci.

Импортировать ПЕРВОЙ строкой в каждом скрипте, до mne и moabb:
    import bci_paths  # noqa: F401

Зачем: по умолчанию MNE и MOABB пишут в домашнюю папку
(~/mne_data — датасеты, ~/.mne/mne-python.json — конфиг).
Здесь всё перенаправляется переменными окружения, конфиг в home не трогаем.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"            # датасеты (в git не идут)
FIGURES = ROOT / "figures"      # картинки для разбора
RESULTS = ROOT / "results"      # цифры экспериментов
_CACHE = ROOT / ".cache"

for d in (DATA, FIGURES, RESULTS, _CACHE / "home", _CACHE / "mpl"):
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(_CACHE / "home"))  # конфиг MNE
os.environ.setdefault("MNE_DATA", str(DATA))
os.environ.setdefault("MNE_DATASETS_BNCI_PATH", str(DATA))         # BCI IV 2a
os.environ.setdefault("MOABB_RESULTS", str(DATA / "moabb_results"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "mpl"))
os.environ.setdefault("LSLAPICFG", str(ROOT / "lsl_api.cfg"))  # LSL: только ошибки, без INFO-болтовни
