#!/bin/bash

# Financial Analysis Agent Setup Script
# This script helps you set up and run the financial analysis system

set -e

echo "=========================================="
echo "Financial Analysis Agent Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if Docker is installed
echo "Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_status "Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_status "Docker Compose is installed"

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

REQUIRED_MODELS=("qwen2-vl:7b" "nomic-embed-text")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        print_status "Model $model is available"
    else
        print_warning "Model $model is not installed"
        MISSING_MODELS+=("$model")
    fi
done

# Check for finance model (may need alternatives)
if ollama list | grep -q "finance-llama-8b"; then
    print_status "Model finance-llama-8b is available"
elif ollama list | grep -q "llama3.1:8b"; then
    print_warning "finance-llama-8b not found, but llama3.1:8b is available (will use as alternative)"
else
    print_warning "Neither finance-llama-8b nor llama3.1:8b found"
    MISSING_MODELS+=("llama3.1:8b")
fi

# Offer to pull missing models
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
    echo "  Please add your financial documents (quarterly reports, annual reports) to:"
    echo "  $(pwd)/data/financials/"
    read -p "Press Enter when you've added the files, or Ctrl+C to exit..."
fi
print_status "Financial documents found"

# Check for valuation parameters
echo ""
echo "Checking for valuation parameters..."
if [ ! -f "data/valuation_parameters.pdf" ]; then
    print_warning "Valuation parameters PDF not found"
    echo "  Please add your valuation parameters PDF to:"
    echo "  $(pwd)/data/valuation_parameters.pdf"
    read -p "Press Enter when you've added the file, or Ctrl+C to exit..."
fi
print_status "Valuation parameters found"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    read -p "Enter company name to analyze: " company_name
    cat > .env << EOF
# Company name to analyze
COMPANY_NAME=${company_name}

# Ollama connection
OLLAMA_BASE_URL=http://host.docker.internal:11434

# Model configurations
VISION_MODEL=qwen2-vl:7b
ANALYSIS_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text
EOF
    print_status ".env file created"
else
    print_status ".env file already exists"
fi

# Build Docker image
echo ""
echo "Building Docker image..."
if docker-compose build; then
    print_status "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Ask if user wants to run now
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the analysis, use:"
echo "  docker-compose up"
echo ""
echo "To run with a different company:"
echo "  COMPANY_NAME=\"Apple Inc\" docker-compose up"
echo ""
read -p "Would you like to run the analysis now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting analysis..."
    docker-compose up
else
    echo ""
    echo "You can start the analysis later with: docker-compose up"
fi

echo ""
print_status "Setup script completed!"