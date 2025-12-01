@echo off
REM Car Damage App Startup Script for Windows

echo.
echo ========================================
echo   Car Damage Estimation System
echo ========================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

echo [OK] Python is installed
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import flask, cv2, numpy, joblib, pandas, ultralytics" > nul 2>&1
if errorlevel 1 (
    echo [!] Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)

echo [OK] All dependencies are installed
echo.

REM Create static directory if it doesn't exist
if not exist "static" (
    mkdir static
    echo [OK] Created static directory
)

echo.
echo ========================================
echo Starting Flask Application...
echo ========================================
echo.
echo The application will be available at:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the Flask app
python app.py

pause
