@echo off
REM Script to install all project dependencies

echo ========================================
echo   DEPENDENCY INSTALLATION
echo ========================================
echo.

echo Installing TensorFlow...
pip install tensorflow==2.15.0

echo.
echo Installing pandas...
pip install pandas==2.1.0

echo.
echo Installing numpy...
pip install numpy==1.24.3

echo.
echo Installing scikit-learn...
pip install scikit-learn==1.3.0

echo.
echo ========================================
echo   INSTALLATION COMPLETE
echo ========================================
echo.

echo Verifying installations...
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import pandas as pd; print('Pandas:', pd.__version__)"
python -c "import numpy as np; print('NumPy:', np.__version__)"
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"

echo.
pause
