# PowerShell startup script for RAG Chatbot
# This ensures the app runs with the correct venv Python

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Starting RAG Chatbot" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv venv"
    Write-Host "Then: .\venv\Scripts\Activate.ps1"
    Write-Host "Then: pip install -r requirements.txt"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Using Python from: venv\Scripts\python.exe" -ForegroundColor Green
Write-Host ""

# Check if Ollama is running
Write-Host "Checking Ollama connection..." -ForegroundColor Yellow
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✅ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "WARNING: Ollama might not be running!" -ForegroundColor Yellow
    Write-Host "Please start Ollama with: ollama serve" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y') {
        exit 1
    }
}

Write-Host ""
Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "The app will open in your default browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Run streamlit with venv Python
& ".\venv\Scripts\python.exe" -m streamlit run app.py


