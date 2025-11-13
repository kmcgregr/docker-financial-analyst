# Financial Analysis Agent Setup Script for Windows PowerShell
# This script helps you set up and run the financial analysis system on Windows

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Financial Analysis Agent Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Function to print colored output
function Print-Status {
    param([string]$Message)
    Write-Host "[✓] $Message" -ForegroundColor Green
}

function Print-Error {
    param([string]$Message)
    Write-Host "[✗] $Message" -ForegroundColor Red
}

function Print-Warning {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

# Check if Docker is installed
Write-Host "Checking prerequisites..."
try {
    $dockerVersion = docker --version 2>$null
    if ($dockerVersion) {
        Print-Status "Docker is installed"
    } else {
        throw
    }
} catch {
    Print-Error "Docker is not installed. Please install Docker Desktop for Windows first."
    Write-Host "Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check if Docker Compose is installed
try {
    $composeVersion = docker-compose --version 2>$null
    if ($composeVersion) {
        Print-Status "Docker Compose is installed"
    } else {
        throw
    }
} catch {
    Print-Error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
}

# Check if Docker is running
try {
    docker ps 2>$null | Out-Null
    Print-Status "Docker is running"
} catch {
    Print-Error "Docker is not running. Please start Docker Desktop."
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

$REQUIRED_MODELS = @("qwen2-vl:7b", "nomic-embed-text")
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
    
    # Check for finance model (may need alternatives)
    if ($ollamaList -match "finance-llama-8b") {
        Print-Status "Model finance-llama-8b is available"
    } elseif ($ollamaList -match "llama3.1:8b") {
        Print-Warning "finance-llama-8b not found, but llama3.1:8b is available (will use as alternative)"
    } else {
        Print-Warning "Neither finance-llama-8b nor llama3.1:8b found"
        $MISSING_MODELS += "llama3.1:8b"
    }
} catch {
    Print-Error "Could not check Ollama models. Ensure Ollama is properly installed."
    exit 1
}

# Offer to pull missing models
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
    Write-Host "  Please add your financial documents (quarterly reports, annual reports) to:" -ForegroundColor Yellow
    Write-Host "  $(Get-Location)\data\financials\" -ForegroundColor Yellow
    Read-Host "Press Enter when you've added the files, or Ctrl+C to exit"
    
    # Re-check after user confirmation
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
    Write-Host "  Please add your valuation parameters PDF to:" -ForegroundColor Yellow
    Write-Host "  $(Get-Location)\data\valuation_parameters.pdf" -ForegroundColor Yellow
    Read-Host "Press Enter when you've added the file, or Ctrl+C to exit"
    
    # Re-check after user confirmation
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
    $companyName = Read-Host "Enter company name to analyze"
    
    $envContent = @"
# Company name to analyze
COMPANY_NAME=$companyName

# Ollama connection
OLLAMA_BASE_URL=http://host.docker.internal:11434

# Model configurations
VISION_MODEL=qwen2-vl:7b
ANALYSIS_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text
"@
    
    $envContent | Out-File -FilePath ".env" -Encoding UTF8
    Print-Status ".env file created"
} else {
    Print-Status ".env file already exists"
}

# Build Docker image
Write-Host ""
Write-Host "Building Docker image..."
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

try {
    docker-compose build 2>&1 | Out-Host
    if ($LASTEXITCODE -eq 0) {
        Print-Status "Docker image built successfully"
    } else {
        throw "Docker build failed"
    }
} catch {
    Print-Error "Failed to build Docker image"
    Print-Error $_.Exception.Message
    exit 1
}

# Ask if user wants to run now
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the analysis, use:" -ForegroundColor Green
Write-Host "  docker-compose up" -ForegroundColor White
Write-Host ""
Write-Host "To run with a different company:" -ForegroundColor Green
Write-Host "  `$env:COMPANY_NAME=`"Apple Inc`"; docker-compose up" -ForegroundColor White
Write-Host ""

$runNow = Read-Host "Would you like to run the analysis now? (y/n)"

if ($runNow -match "^[Yy]") {
    Write-Host ""
    Write-Host "Starting analysis..." -ForegroundColor Cyan
    Write-Host "This may take 10-15 minutes depending on document complexity..." -ForegroundColor Yellow
    Write-Host ""
    
    try {
        docker-compose up
    } catch {
        Print-Error "Error running analysis"
        Print-Error $_.Exception.Message
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "You can start the analysis later with: docker-compose up" -ForegroundColor Yellow
}

Write-Host ""
Print-Status "Setup script completed!"
Write-Host ""
Write-Host "Generated reports will be saved to: $(Get-Location)\data\output\" -ForegroundColor Green