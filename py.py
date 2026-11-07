"""Запуск скриптов проекта базовым Python с пакетами из .venv.

Нужен, если Windows (Smart App Control) блокирует .venv/Scripts/python.exe.
Запуск (из projects/bci):
    .python/cpython-3.12.16-windows-x86_64-none/python.exe -X utf8 py.py eegnet.py cross
"""

import runpy
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / ".venv" / "Lib" / "site-packages"))
sys.path.insert(0, str(ROOT))
script, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(str(ROOT / script), run_name="__main__")
