@echo off
REM Start the web demo locally (double-click)
REM Requires Python 3.11 (TensorFlow 2.15 is not compatible with 3.12+)
REM NOTE: avoid parentheses inside if-blocks - cmd mis-parses them.

cd /d "%~dp0.."

echo ========================================
echo   FOOTBALL RESULT PREDICTOR - WEB DEMO
echo ========================================
echo.

echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH.
    echo Install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 is required. TensorFlow 2.15 does not support 3.12 or newer.
    echo Current interpreter:
    python --version
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python -c "import tensorflow, fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt -r requirements-dev.txt
    if %errorlevel% neq 0 (
        echo ERROR installing dependencies.
        pause
        exit /b 1
    )
)

echo.
echo Checking artifacts...
if not exist "artifacts\production_model.keras" (
    echo Building artifacts. This requires trained models in models/ and data/.
    python src\build_artifacts.py
    if %errorlevel% neq 0 (
        echo ERROR building artifacts.
        pause
        exit /b 1
    )
)

echo.
echo Opening browser at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000

echo Starting server. Press Ctrl+C to stop.
python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000

pause
