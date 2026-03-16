# Setup script for ReturnShield AI (Windows)
# This script automates the installation and configuration process

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "ReturnShield AI - Setup Script" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "✓ Checking Python version..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1 | Select-String -Pattern "(\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches[0].Value }
    Write-Host "  Python $pythonVersion detected ✓" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Create virtual environment
Write-Host "✓ Creating virtual environment..." -ForegroundColor Green
if (Test-Path ".venv") {
    Write-Host "  Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "  Virtual environment created ✓" -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "✓ Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1
Write-Host "  Virtual environment activated ✓" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "✓ Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip --quiet
Write-Host "  pip upgraded ✓" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "✓ Installing dependencies..." -ForegroundColor Green
pip install -r requirements.txt --quiet
Write-Host "  Dependencies installed ✓" -ForegroundColor Green
Write-Host ""

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "✓ Creating .env file..." -ForegroundColor Green
    Copy-Item ".env.example" ".env"
    Write-Host "  .env file created ✓" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Edit .env file with your AWS credentials!" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "  .env file already exists" -ForegroundColor Yellow
}

# Create data directory if it doesn't exist
if (-not (Test-Path "data")) {
    Write-Host "✓ Creating data directory..." -ForegroundColor Green
    New-Item -ItemType Directory -Path "data" | Out-Null
    Write-Host "  data directory created ✓" -ForegroundColor Green
}

# Check AWS credentials
Write-Host "✓ Checking AWS configuration..." -ForegroundColor Green
if (Get-Command aws -ErrorAction SilentlyContinue) {
    try {
        aws sts get-caller-identity | Out-Null
        Write-Host "  AWS credentials configured ✓" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️  AWS credentials not configured or invalid" -ForegroundColor Yellow
        Write-Host "     Run: aws configure" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️  AWS CLI not installed" -ForegroundColor Yellow
    Write-Host "     Install: pip install awscli" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  1. Edit .env with your AWS credentials"
Write-Host "  2. Run the pipeline: python -m src.main"
Write-Host "  3. Start API server: python -m uvicorn api.server:app --reload"
Write-Host "  4. Launch dashboard: streamlit run dashboard/app.py"
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Green
Write-Host "  - README.md - Quick start and overview"
Write-Host "  - ARCHITECTURE.md - System design details"
Write-Host "  - DEPLOYMENT.md - Production deployment guide"
Write-Host ""
Write-Host "For help, visit: https://github.com/yourorg/return-integrity" -ForegroundColor Cyan
Write-Host ""
