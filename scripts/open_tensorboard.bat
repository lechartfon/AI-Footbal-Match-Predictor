@echo off
REM Quick script to open TensorBoard (Windows compatible)

echo ========================================
echo   OPENING TENSORBOARD
echo ========================================
echo.
echo Starting TensorBoard server...
echo Open your browser at: http://localhost:6006
echo.
echo Press Ctrl+C to stop
echo.

REM Check that TensorBoard is installed in this Python
python -m pip show tensorboard >nul 2>&1
if %errorlevel% neq 0 (
    echo TensorBoard is not installed. Installing...
    python -m pip install tensorboard
)

REM Run TensorBoard using the same Python interpreter
python -m tensorboard.main --logdir=../logs --port=6006
