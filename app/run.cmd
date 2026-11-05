@echo off
rem Двойной клик: запускает локальный сервер и открывает визуализатор BCI.
rem Окно сервера не закрывать, пока смотришь. Закрыл окно — сервер остановлен.
chcp 65001 >nul
cd /d "%~dp0"
start "BCI server - do not close" python -m http.server 8765 --bind 127.0.0.1
timeout /t 2 /nobreak >nul
start "" http://localhost:8765/stream.html
