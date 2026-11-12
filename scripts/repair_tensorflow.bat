@echo off
REM Script to REPAIR corrupted TensorFlow installation

echo ========================================
echo   TENSORFLOW REPAIR
echo ========================================
echo.

echo STEP 1: Uninstalling corrupted TensorFlow...
pip uninstall -y tensorflow
pip uninstall -y tensorflow-intel
pip uninstall -y tensorflow-cpu
pip uninstall -y keras
pip uninstall -y keras-preprocessing

echo.
echo STEP 2: Cleaning pip cache...
pip cache purge

echo.
echo STEP 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo STEP 4: Installing TensorFlow 2.15.0...
pip install tensorflow==2.15.0 --no-cache-dir

echo.
echo STEP 5: Installing standalone Keras (backup)...
pip install keras==2.15.0

echo.
echo ========================================
echo   VERIFICATION
echo ========================================
echo.

echo Verifying installation...
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "from tensorflow import keras; print('Keras integrated: OK')"
python -c "from tensorflow.keras.layers import Dense; print('Layers available: OK')"

echo.
echo ========================================
echo   REPAIR COMPLETE
echo ========================================
echo.
pause
