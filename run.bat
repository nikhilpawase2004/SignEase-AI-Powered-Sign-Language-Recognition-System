@echo off
echo =================================================
echo   SignVerse - Unified Sign Language Detection    
echo =================================================
echo.

:: Activate the conda environment and run
call conda activate sign_language_unified
if errorlevel 1 (
    echo ERROR: Could not activate conda environment 'sign_language_unified'
    echo Please run: conda env create -f environment.yml
    pause
    exit /b 1
)

echo Environment activated: sign_language_unified
echo Starting Flask server...
echo Open http://localhost:5000 in your browser
echo.

python app.py

pause
