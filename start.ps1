# MedRAG Quick Start Script
# Run this from C:\Users\nitro\Medchat

Write-Host "=== MedRAG Startup ===" -ForegroundColor Cyan

# Check Ollama is running
Write-Host "`n[1/2] Checking Ollama..." -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    Write-Host "      Ollama is running ✅" -ForegroundColor Green
} catch {
    Write-Host "      Ollama not running. Starting it..." -ForegroundColor Red
    Write-Host "      Please open a NEW terminal and run: ollama serve" -ForegroundColor Red
    Write-Host "      Then re-run this script." -ForegroundColor Red
    exit 1
}

# Start FastAPI server
Write-Host "`n[2/2] Starting MedRAG FastAPI server..." -ForegroundColor Yellow
Write-Host "      URL: http://localhost:8000" -ForegroundColor Green
Write-Host "      API: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "`n      Press Ctrl+C to stop the server`n" -ForegroundColor Gray

.\venv\Scripts\python.exe app.py
