# Launch the Breast Cancer ML web application (Streamlit).
# Usage:  ./run_app.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Host "Starting Streamlit app on http://localhost:8501 ..." -ForegroundColor Green
python -m streamlit run app.py --server.port 8501
