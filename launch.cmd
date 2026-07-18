@echo off
REM LiveCap Local launcher (task 6).
REM   1. Activates the project's .venv (uv-created, Python 3.12, torch cu128, moshi, PyQt6).
REM   2. Sets NO_TORCH_COMPILE=1 (defense in depth: livecap/asr.py and main.py already
REM      set it via os.environ.setdefault, but Triton's absence on Windows means this
REM      must never depend on a single code path -- see CONTRACT.md gotchas).
REM   3. Verifies Ollama answers on localhost:11434, starts it if not (best-effort:
REM      main.py's own livecap.translator.ensure_ollama_running() is the authoritative
REM      check/retry and will still raise a clear error if this fails).
REM   4. Launches main.py with any extra arguments passed through unchanged
REM      (e.g. launch.cmd --input-wav tests\fixtures\test_fr.wav --no-overlay).
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [launch.cmd] .venv introuvable. Voir README.md section Installation ^(uv venv --python 3.12 .venv^).
    exit /b 1
)

set "NO_TORCH_COMPILE=1"

echo [launch.cmd] Verification d'Ollama sur localhost:11434...
curl -s -o NUL -w "%%{http_code}" http://localhost:11434/api/tags > "%TEMP%\livecap_ollama_status.txt" 2>NUL
set /p OLLAMA_STATUS=<"%TEMP%\livecap_ollama_status.txt"
del "%TEMP%\livecap_ollama_status.txt" >NUL 2>&1

if "%OLLAMA_STATUS%"=="200" (
    echo [launch.cmd] Ollama deja demarre.
) else (
    echo [launch.cmd] Ollama ne repond pas, tentative de demarrage ^(ollama serve^)...
    where ollama >NUL 2>&1
    if errorlevel 1 (
        echo [launch.cmd] AVERTISSEMENT: 'ollama' introuvable sur le PATH. Installer https://ollama.com
        echo [launch.cmd] main.py tentera quand meme de demarrer/joindre Ollama au lancement.
    ) else (
        start "ollama serve" /min ollama serve
        timeout /t 3 /nobreak >NUL
    )
)

echo [launch.cmd] Lancement de main.py %*
".venv\Scripts\python.exe" main.py %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
