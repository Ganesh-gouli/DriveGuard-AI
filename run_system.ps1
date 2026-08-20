$ErrorActionPreference = "Stop"
Write-Host "Starting Driver AI Backend..." -ForegroundColor Green
Start-Process python -ArgumentList "backend/app.py" -WorkingDirectory "$PSScriptRoot" -WindowStyle Normal
Write-Host "Waiting for backend to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 5
Write-Host "Starting Detection System..." -ForegroundColor Green
python detection/detect.py
Write-Host "Driver AI stopped." -ForegroundColor Yellow
