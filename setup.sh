#!/bin/bash

# Financial Analysis Agent Setup Script
# Sets up a Python virtual environment and validates prerequisites.

set -e

echo "=========================================="
echo "Financial Analysis Agent Setup"
echo "=========================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error()  { echo -e "${RED}[✗]${NC} $1"; }
print_warning(){ echo -e "${YELLOW}[!]${NC} $1"; }

# Check Python
echo "Checking prerequisites..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi
print_status "Python is installed ($(python3 --version 2>&1))"

# Check if Ollama is running
echo ""
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_error "Ollama is not running. Please start Ollama first."
    exit 1
fi
print_status "Ollama is running"

# Check for required models
echo ""
echo "Checking required Ollama models..."

REQUIRED_MODELS=("qwen2.5vl:7b" "nomic-embed-text")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        print_status "Model $model is available"
    else
        print_warning "Model $model is not installed"
        MISSING_MODELS+=("$model")
    fi
done

ANALYSIS_MODEL="gpt-oss:20b"
if ollama list | grep -q "$ANALYSIS_MODEL"; then
    print_status "Model $ANALYSIS_MODEL is available"
elif ollama list | grep -q "llama3.1:8b"; then
    print_warning "$ANALYSIS_MODEL not found, but llama3.1:8b is available (update ANALYSIS_MODEL in .env)"
else
    print_warning "Neither $ANALYSIS_MODEL nor llama3.1:8b found"
    MISSING_MODELS+=("llama3.1:8b")
fi

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    print_warning "Missing models: ${MISSING_MODELS[*]}"
    read -p "Would you like to pull missing models now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for model in "${MISSING_MODELS[@]}"; do
            echo "Pulling $model..."
            ollama pull "$model"
        done
    else
        print_error "Cannot proceed without required models. Exiting."
        exit 1
    fi
fi

# Create directory structure
echo ""
echo "Setting up directory structure..."
mkdir -p data/financials
mkdir -p data/output
print_status "Directories created"

# Check for financial documents
echo ""
echo "Checking for financial documents..."
if [ -z "$(ls -A data/financials/*.pdf 2>/dev/null)" ]; then
    print_warning "No PDF files found in data/financials/"
    echo "  Add financial documents (quarterly reports, annual reports) to:"
    echo "  $(pwd)/data/financials/"
    read -p "Press Enter when ready, or Ctrl+C to exit..."
fi
print_status "Financial documents checked"

# Check for valuation parameters
echo ""
echo "Checking for valuation parameters..."
if [ ! -f "data/valuation_parameters.pdf" ]; then
    print_warning "Valuation parameters PDF not found"
    echo "  Place your valuation parameters PDF at:"
    echo "  $(pwd)/data/valuation_parameters.pdf"
    read -p "Press Enter when ready, or Ctrl+C to exit..."
fi
print_status "Valuation parameters checked"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    read -p "Enter company name to analyze (optional — auto-detected from PDFs): " company_name
    cat > .env << EOF
# Company name to analyze (optional — auto-detected if omitted)
COMPANY_NAME=${company_name:-}

# Ollama connection
OLLAMA_BASE_URL=http://localhost:11434

# Model configurations
VISION_MODEL=qwen2.5vl:7b
ANALYSIS_MODEL=gpt-oss:20b
EMBEDDING_MODEL=nomic-embed-text
EOF
    print_status ".env file created"
else
    print_status ".env file already exists"
fi

# Set up virtual environment
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

source venv/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
print_status "Dependencies installed"

# Ask if user wants to run now
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the analysis:"
echo "  source venv/bin/activate"
echo "  python -m main"
echo ""
echo "To run the web interface:"
echo "  source venv/bin/activate"
echo "  uvicorn web.main:app --reload"
echo ""
read -p "Would you like to run the analysis now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting analysis..."
    python -m main
else
    echo ""
    echo "Run later with: source venv/bin/activate && python -m main"
fi

echo ""
print_status "Setup script completed!"
