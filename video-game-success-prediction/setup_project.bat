@echo off
title Video Game Prediction - Quick Setup
color 0B

echo.
echo ========================================
echo   Video Game Success Prediction
echo   First-Time Setup
echo ========================================
echo.

cd /d "e:\SLIIT\Academic\3y\MINI project\New folder\video-game-success-prediction"

echo [1/3] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment!
        echo Make sure Python is installed and in PATH
        pause
        exit /b 1
    )
    echo Virtual environment created!
) else (
    echo Virtual environment already exists!
)

echo [2/3] Installing dependencies...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo [3/3] Training initial model...
.\.venv\Scripts\python.exe -m src.train_model
if %errorlevel% neq 0 (
    echo WARNING: Model training failed - you may need to add your dataset first
    echo Place vg_sales_2024.csv in the data\ folder
) else (
    echo Model trained successfully!
)

echo.
echo ========================================
echo   Setup Complete!
echo   
echo   Next steps:
echo   1. Place your vg_sales_2024.csv in data\ folder
echo   2. Double-click run_project.bat to start the app
echo ========================================
echo.
pause