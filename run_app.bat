@echo off
REM Startup script for RAG Chatbot
REM This ensures the app runs with the correct venv Python

echo ============================================
echo Starting RAG Chatbot
echo ============================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

echo Using Python from: venv\Scripts\python.exe
echo.

REM Check if Ollama is running
echo Checking Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Ollama might not be running!
    echo Please start Ollama with: ollama serve
    echo.
    pause
)

echo.
echo Starting Streamlit app...
echo The app will open in your default browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ============================================
echo.

REM Run streamlit with venv Python
venv\Scripts\python.exe -m streamlit run app.py

pause


