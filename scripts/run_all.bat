@echo off
REM Script to run the entire project automatically
REM Windows Batch Script

echo ========================================
echo   FOOTBALL RESULT PREDICTOR
echo   Automatic Execution
echo ========================================
echo.

echo Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
pip show tensorflow >nul 2>&1
if %errorlevel% neq 0 (
    echo TensorFlow not installed. Installing dependencies...
    pip install -r ../requirements.txt
)

echo.
echo ========================================
echo   PHASE 1: DATA PREPARATION
echo ========================================
cd ../src
python data_preparation.py
if %errorlevel% neq 0 (
    echo ERROR in data preparation
    cd ..
    pause
    exit /b 1
)

echo.
echo ========================================
echo   PHASE 2: MODEL VERIFICATION
echo ========================================
python model_architecture.py

echo.
echo ========================================
echo   PHASE 3-4: TRAINING
echo ========================================
echo IMPORTANT: This will take 10-30 minutes
echo.
set /p continue="Continue with training? (Y/N): "
if /i "%continue%" neq "Y" (
    echo Training cancelled
    cd ..
    pause
    exit /b 0
)

python train_experiments.py
if %errorlevel% neq 0 (
    echo ERROR in training
    cd ..
    pause
    exit /b 1
)

echo.
echo ========================================
echo   PHASE 5: TEST EVALUATION
echo ========================================
python evaluate_test.py

echo.
echo ========================================
echo   PROJECT COMPLETE
echo ========================================
echo.
echo To use the interactive predictor:
echo   cd ../src
echo   python predict_interactive.py
echo.
echo To open TensorBoard:
echo   tensorboard --logdir=../logs
echo   Then open: http://localhost:6006
echo.

cd ..
pause
