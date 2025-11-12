# Quick Start Guide

Get your financial analysis system up and running in 5 minutes.

## Prerequisites Checklist

- [ ] Docker installed
- [ ] Docker Compose installed
- [ ] Ollama installed and running
- [ ] Financial PDF documents ready
- [ ] Valuation parameters PDF ready

## Step-by-Step Setup

### 1. Install Ollama Models (5-10 minutes)

```bash
# Install required models
ollama pull qwen2-vl:7b
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Verify installation
ollama list
```

### 2. Set Up Project Structure

```bash
# Create project directory
mkdir financial-analyst
cd financial-analyst

# Create all necessary files (copy from artifacts above)
# - financial_analysis.py
# - Dockerfile
# - docker-compose.yml
# - requirements.txt
# - .env.example
# - setup.sh

# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

### 3. Add Your Documents

```bash
# Add financial documents
cp /path/to/your/Q*.pdf data/financials/
cp /path/to/your/annual_report.pdf data/financials/

# Add valuation parameters
cp /path/to/valuation_params.pdf data/valuation_parameters.pdf
```

### 4. Configure Company Name

```bash
# Edit .env file
nano .env

# Set your company name
COMPANY_NAME=Apple Inc
```

### 5. Run Analysis

```bash
# Build and run
docker-compose up

# Or run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### 6. Get Your Report

```bash
# Reports are saved in data/output/
ls -lh data/output/

# View the latest report
cat data/output/investment_report_*.txt
```

## Quick Commands

```bash
# Run analysis for specific company
COMPANY_NAME="Microsoft" docker-compose up

# Stop the container
docker-compose down

# Rebuild after code changes
docker-compose build --no-cache

# View real-time logs
docker-compose logs -f financial-analyst

# Clean up everything
docker-compose down -v
rm -rf data/output/*
```

## Troubleshooting Quick Fixes

### Can't connect to Ollama?
```bash
# Test Ollama connectivity
curl http://localhost:11434/api/tags

# If fails, restart Ollama
# On macOS/Linux: restart the Ollama app
# On Windows: restart Ollama from system tray
```

### Models not found?
```bash
# Re-pull models
ollama pull qwen2-vl:7b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### Container can't reach host?
```bash
# Test Docker host connectivity
docker run --rm alpine ping -c 3 host.docker.internal

# On Linux, you may need to use:
# docker run --rm --add-host=host.docker.internal:host-gateway alpine ping -c 3 host.docker.internal
```

### Out of memory?
```bash
# Use smaller models
ollama pull qwen2-vl:2b  # Instead of 7b

# Or increase Docker memory in Docker Desktop settings
```

## File Structure at a Glance

```
financial-analyst/
├── financial_analysis.py          ← Main app
├── Dockerfile                      ← Docker config
├── docker-compose.yml              ← Docker Compose
├── requirements.txt                ← Python deps
├── .env                           ← Your settings
├── setup.sh                       ← Setup script
└── data/
    ├── financials/                ← Put PDFs here
    │   ├── Q1_2024.pdf
    │   └── annual_2023.pdf
    ├── valuation_parameters.pdf   ← Your valuation guide
    └── output/                    ← Reports appear here
        └── investment_report_*.txt
```

## What Happens During Analysis?

1. **Extract** (1-2 min): Vision model reads all PDFs
2. **Analyze Business** (2-3 min): Understand business model
3. **Analyze Growth** (2-3 min): Calculate metrics and KPIs
4. **Value Company** (2-3 min): Apply valuation methods
5. **Recommend** (1-2 min): Final investment opinion
6. **Generate Report** (<1 min): Create formatted report

**Total time**: ~10-15 minutes depending on document size

## Expected Output

Your report will include:

```
===============================================================================
FINANCIAL ANALYSIS & INVESTMENT REPORT
===============================================================================

Company: Apple Inc
Report Generated: 2024-10-29 14:30:22

[Executive Summary]
[Business Overview]
[Financial Metrics]
[Growth Analysis]
[Valuation]
[Investment Recommendation: BUY/HOLD/SELL]
[Risk Factors]

===============================================================================
```

## Next Steps After First Run

1. **Review the report** in `data/output/`
2. **Adjust settings** in `.env` if needed
3. **Add more documents** for deeper analysis
4. **Customize agents** in `financial_analysis.py`
5. **Run comparative analysis** on multiple companies

## Tips for Best Results

✅ **DO:**
- Use clear, high-quality PDFs
- Include multiple periods (quarterly + annual)
- Provide detailed valuation parameters
- Review and validate AI outputs

❌ **DON'T:**
- Use scanned images without OCR
- Mix documents from different companies
- Trust AI recommendations blindly
- Skip reviewing the raw data extraction

## Getting Help

- Check main README.md for detailed documentation
- Review CrewAI docs: https://docs.crewai.com
- Check Ollama docs: https://ollama.ai/docs
- Verify Docker setup: https://docs.docker.com

## One-Line Complete Setup

```bash
# Complete setup in one command (after installing prerequisites)
git clone <repo> && cd financial-analyst && ./setup.sh && docker-compose up
```

That's it! You're ready to analyze companies like a pro. 🚀