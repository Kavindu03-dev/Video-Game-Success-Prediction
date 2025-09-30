@echo off
title Video Game Success Prediction - Project Launcher
color 0A

echo.
echo ========================================
echo   Video Game Success Prediction
echo   Project Launcher
echo ========================================
echo.

cd /d "e:\SLIIT\Academic\3y\MINI project\New folder\video-game-success-prediction"

echo [1/4] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    echo Then install requirements: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [2/4] Checking model file...
if not exist "model.pkl" (
    echo Model not found. Training new model...
    .\.venv\Scripts\python.exe -m src.train_model
    if %errorlevel% neq 0 (
        echo ERROR: Model training failed!
        pause
        exit /b 1
    )
    echo Model trained successfully!
) else (
    echo Model found: model.pkl
)

echo [3/4] Checking dataset...
if exist "data\vg_sales_2024.csv" (
    echo Dataset found: data\vg_sales_2024.csv
) else if exist "data\raw\vg_sales_2024.csv" (
    echo Dataset found: data\raw\vg_sales_2024.csv
) else (
    echo WARNING: Dataset not found!
    echo Please place your vg_sales_2024.csv in data\ or data\raw\ folder
    echo.
    echo Continuing anyway - app will show sample data...
)

echo [4/4] Starting Streamlit app...
echo.
echo ========================================
echo   App is starting...
echo   Open your browser to: http://localhost:8504
echo   Press Ctrl+C to stop the app
echo ========================================
echo.

.\.venv\Scripts\python.exe -m streamlit run app\app.py --server.port 8504

echo.
echo App stopped.
pause