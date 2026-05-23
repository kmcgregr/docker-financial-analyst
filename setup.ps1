# Financial Analysis Agent Setup Script for Windows PowerShell
# Sets up a Python virtual environment and validates prerequisites.

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Financial Analysis Agent Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

function Print-Status { param([string]$M); Write-Host "[✓] $M" -ForegroundColor Green }
function Print-Error  { param([string]$M); Write-Host "[✗] $M" -ForegroundColor Red }
function Print-Warning{ param([string]$M); Write-Host "[!] $M" -ForegroundColor Yellow }

# Check Python
Write-Host "Checking prerequisites..."
try {
    $pyVersion = python --version 2>&1
    if ($pyVersion -match "Python 3\.(1[1-9]|[2-9]\d)") {
        Print-Status "Python is installed ($pyVersion)"
    } else {
        Print-Warning "Python 3.11+ is recommended. Found: $pyVersion"
    }
} catch {
    Print-Error "Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
}

# Check if Ollama is running
Write-Host ""
Write-Host "Checking Ollama..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5 -ErrorAction Stop
    Print-Status "Ollama is running"
} catch {
    Print-Error "Ollama is not running. Please start Ollama first."
    Write-Host "Download from: https://ollama.ai/download/windows" -ForegroundColor Yellow
    exit 1
}

# Check for required models
Write-Host ""
Write-Host "Checking required Ollama models..."

$REQUIRED_MODELS = @("qwen2.5vl:7b", "nomic-embed-text")
$MISSING_MODELS = @()

try {
    $ollamaList = ollama list 2>$null

    foreach ($model in $REQUIRED_MODELS) {
        if ($ollamaList -match $model) {
            Print-Status "Model $model is available"
        } else {
            Print-Warning "Model $model is not installed"
            $MISSING_MODELS += $model
        }
    }

    $ANALYSIS_MODEL = "gpt-oss:20b"
    if ($ollamaList -match [regex]::Escape($ANALYSIS_MODEL)) {
        Print-Status "Model $ANALYSIS_MODEL is available"
    } elseif ($ollamaList -match "llama3.1:8b") {
        Print-Warning "$ANALYSIS_MODEL not found, but llama3.1:8b is available (update ANALYSIS_MODEL in .env)"
    } else {
        Print-Warning "Neither $ANALYSIS_MODEL nor llama3.1:8b found"
        $MISSING_MODELS += "llama3.1:8b"
    }
} catch {
    Print-Error "Could not check Ollama models. Ensure Ollama is properly installed."
    exit 1
}

if ($MISSING_MODELS.Count -gt 0) {
    Write-Host ""
    Print-Warning "Missing models: $($MISSING_MODELS -join ', ')"
    $response = Read-Host "Would you like to pull missing models now? (y/n)"

    if ($response -match "^[Yy]") {
        foreach ($model in $MISSING_MODELS) {
            Write-Host "Pulling $model..." -ForegroundColor Cyan
            ollama pull $model
            if ($LASTEXITCODE -ne 0) {
                Print-Error "Failed to pull $model"
                exit 1
            }
        }
    } else {
        Print-Error "Cannot proceed without required models. Exiting."
        exit 1
    }
}

# Create directory structure
Write-Host ""
Write-Host "Setting up directory structure..."
$directories = @("data\financials", "data\output")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Print-Status "Directories created"

# Check for financial documents
Write-Host ""
Write-Host "Checking for financial documents..."
$pdfFiles = Get-ChildItem -Path "data\financials\*.pdf" -ErrorAction SilentlyContinue

if ($pdfFiles.Count -eq 0) {
    Print-Warning "No PDF files found in data\financials\"
    Write-Host "  Add financial documents (quarterly reports, annual reports) to:" -ForegroundColor Yellow
    Write-Host "  $(Get-Location)\data\financials\" -ForegroundColor Yellow
    Read-Host "Press Enter when ready, or Ctrl+C to exit"

    $pdfFiles = Get-ChildItem -Path "data\financials\*.pdf" -ErrorAction SilentlyContinue
    if ($pdfFiles.Count -eq 0) {
        Print-Error "Still no PDF files found. Please add documents and run setup again."
        exit 1
    }
}
Print-Status "Financial documents found ($($pdfFiles.Count) PDF files)"

# Check for valuation parameters
Write-Host ""
Write-Host "Checking for valuation parameters..."
if (-not (Test-Path "data\valuation_parameters.pdf")) {
    Print-Warning "Valuation parameters PDF not found"
    Write-Host "  Place your valuation parameters PDF at:" -ForegroundColor Yellow
    Write-Host "  $(Get-Location)\data\valuation_parameters.pdf" -ForegroundColor Yellow
    Read-Host "Press Enter when ready, or Ctrl+C to exit"

    if (-not (Test-Path "data\valuation_parameters.pdf")) {
        Print-Error "Valuation parameters PDF still not found. Please add it and run setup again."
        exit 1
    }
}
Print-Status "Valuation parameters found"

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "Creating .env file..."
    $companyName = Read-Host "Enter company name to analyze (optional — auto-detected from PDFs)"

    $envContent = @"
# Company name to analyze (optional — auto-detected if omitted)
COMPANY_NAME=$companyName

# Ollama connection
OLLAMA_BASE_URL=http://localhost:11434

# Model configurations
VISION_MODEL=qwen2.5vl:7b
ANALYSIS_MODEL=gpt-oss:20b
EMBEDDING_MODEL=nomic-embed-text
"@

    $envContent | Out-File -FilePath ".env" -Encoding UTF8
    Print-Status ".env file created"
} else {
    Print-Status ".env file already exists"
}

# Set up virtual environment
Write-Host ""
Write-Host "Setting up Python virtual environment..."
if (-not (Test-Path "venv")) {
    python -m venv venv
    Print-Status "Virtual environment created"
} else {
    Print-Status "Virtual environment already exists"
}

& .\venv\Scripts\pip install -r requirements.txt --quiet
Print-Status "Dependencies installed"

# Ask if user wants to run now
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the analysis:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\activate; python -m main" -ForegroundColor White
Write-Host ""
Write-Host "To run the web interface:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\activate; uvicorn web.main:app --reload" -ForegroundColor White
Write-Host ""

$runNow = Read-Host "Would you like to run the analysis now? (y/n)"

if ($runNow -match "^[Yy]") {
    Write-Host ""
    Write-Host "Starting analysis..." -ForegroundColor Cyan
    Write-Host "This may take 10-15 minutes depending on document complexity..." -ForegroundColor Yellow
    Write-Host ""

    try {
        & .\venv\Scripts\python -m main
    } catch {
        Print-Error "Error running analysis"
        Print-Error $_.Exception.Message
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "Run later with: .\venv\Scripts\activate; python -m main" -ForegroundColor Yellow
}

Write-Host ""
Print-Status "Setup script completed!"
Write-Host ""
Write-Host "Generated reports will be saved to: $(Get-Location)\data\output\" -ForegroundColor Green
