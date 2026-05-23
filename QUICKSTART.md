# Quick Start Guide

Get your financial analysis system up and running in 5 minutes.

## Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] Ollama installed and running
- [ ] Financial PDF documents ready
- [ ] Valuation parameters PDF ready

## Step-by-Step Setup

### 1. Install Ollama Models (5-10 minutes)

```bash
# Install required models
ollama pull qwen2.5vl:7b
ollama pull gpt-oss:20b
ollama pull nomic-embed-text

# Verify installation
ollama list
```

### 2. Set Up Python Virtual Environment

```bash
# Navigate to project directory
cd docker-financial-analyst

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add Your Documents

```bash
# Add financial documents
cp /path/to/your/Q*.pdf data/financials/
cp /path/to/your/annual_report.pdf data/financials/

# Add valuation parameters
cp /path/to/valuation_params.pdf data/valuation_parameters.pdf
```

### 4. Run Analysis

```bash
# Company name is auto-extracted from documents
python -m main

# Or override with an explicit name
python -m main --company-name "Apple Inc"
```

### 5. Get Your Report

```bash
# Reports are saved in data/output/
ls -lh data/output/

# View the latest report
cat data/output/investment_report_*.md
```

## Quick Commands

```bash
# Activate virtual environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate          # macOS/Linux

# Run analysis
python -m main

# Run with explicit company name
python -m main --company-name "Microsoft"

# Run web interface (in another terminal)
uvicorn web.main:app --reload

# Run tests
python -m pytest tests/ -v

# Clean workspace
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
ollama pull qwen2.5vl:7b
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

### Module not found errors?
```bash
# Ensure virtual environment is activated
# Ensure all deps are installed
pip install -r requirements.txt
```

## File Structure at a Glance

```
docker-financial-analyst/
├── main.py                  ← CLI entry point
├── orchestrator.py          ← Workflow coordinator
├── agents.py                ← AI agent definitions
├── tasks.py                 ← Pipeline task definitions
├── vision_extractor.py      ← PDF extraction
├── valuation_rag.py         ← Valuation parameter RAG
├── utils.py                 ← Utility functions
├── requirements.txt         ← Python deps
├── .env                     ← Your settings
├── setup.sh / setup.ps1     ← Setup scripts
└── data/
    ├── financials/          ← Put PDFs here
    │   ├── Q1_2024.pdf
    │   └── annual_2023.pdf
    ├── valuation_parameters.pdf
    └── output/              ← Reports appear here
        └── investment_report_*.md
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
4. **Run comparative analysis** on multiple companies

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
