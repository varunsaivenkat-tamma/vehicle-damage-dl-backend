# Car Damage App Startup Script for PowerShell

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Car Damage Estimation System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python is installed: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.9 or higher" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import flask, cv2, numpy, joblib, pandas, ultralytics" 2>&1 | Out-Null
    Write-Host "[OK] All dependencies are installed" -ForegroundColor Green
} catch {
    Write-Host "[!] Installing required packages..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install requirements" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[OK] Dependencies installed successfully" -ForegroundColor Green
}

Write-Host ""

# Create static directory if it doesn't exist
if (-not (Test-Path "static")) {
    New-Item -ItemType Directory -Path "static" -Force | Out-Null
    Write-Host "[OK] Created static directory" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Flask Application..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The application will be available at:" -ForegroundColor Yellow
Write-Host "http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the Flask app
python app.py
