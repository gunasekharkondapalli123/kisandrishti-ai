@echo off
title KisanDrishti AI - Andhra Pradesh Agro Assistant
echo ===================================================
echo     🌾 KisanDrishti AI (కిసాన్ దృష్టి)
echo   Edge AI Crop Diagnosis & APMC Mandi Intelligence
echo ===================================================
echo.
echo Starting local KisanDrishti server...
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo Opening browser at http://localhost:8501 ...
start http://localhost:8501

streamlit run app.py
pause
