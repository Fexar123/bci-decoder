"""Скачать Dreyer2023 (87 человек) в data/ — по одному, с замером размера и времени.

Запуск (из projects/bci):
    .venv/Scripts/python dl_dreyer.py 1 1      # только первый — замер
    .venv/Scripts/python dl_dreyer.py 2 87     # остальные
Останавливается сам, если на диске C: меньше 8 ГБ свободного места.
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import shutil
import sys
import time
import warnings

import mne
from moabb.datasets import Dreyer2023

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
MIN_FREE_GB = 8


def folder_gb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main():
    first, last = int(sys.argv[1]), int(sys.argv[2])
    ds = Dreyer2023()
    t_all = time.time()
    for s in range(first, last + 1):
        free = shutil.disk_usage(bci_paths.ROOT).free / 1e9
        if free < MIN_FREE_GB:
            print(f"стоп: на диске {free:.1f} ГБ свободно, меньше {MIN_FREE_GB}", flush=True)
            break
        t0 = time.time()
        try:
            ds.get_data(subjects=[s])
        except Exception as e:   # битый файл у одного человека не должен ронять всё
            print(f"S{s}: ошибка — {e}", flush=True)
            continue
        print(f"S{s}: {time.time() - t0:.0f} с · data/ = {folder_gb(bci_paths.DATA):.2f} ГБ · "
              f"свободно {free:.1f} ГБ", flush=True)
    print(f"всего {time.time() - t_all:.0f} с", flush=True)


if __name__ == "__main__":
    main()
