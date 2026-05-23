# Financial Analyst

Multi-agent AI system that analyzes financial PDF documents and generates structured investment reports using local LLMs via [Ollama](https://ollama.ai).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Required Ollama models (configured in `.env`):
  - Vision: `qwen2.5vl:7b`
  - Analysis: `gpt-oss:20b` (or `llama3.1:8b` as alternative)
  - Embedding: `nomic-embed-text`

## Quick Start

```bash
# 1. Setup environment (validates prerequisites, creates dirs)
.\setup.ps1         # Windows
./setup.sh          # Linux/macOS

# 2. Activate the virtual environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate          # Linux/macOS

# 3. Add PDF reports to data/financials/

# 4. Run analysis (company name is auto-extracted)
python -m main
```

Reports are written to `data/output/`.

## Alternative: Run Without Setup Script

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate          # Linux/macOS
pip install -r requirements.txt
python -m main
```

## Web Interface

```bash
uvicorn web.main:app --reload
# Open http://localhost:8000
```

## Architecture

Five specialized agents run sequentially:

1. **Document Analyst** — Extracts financial data from PDFs
2. **Business Analyst** — Analyzes business model and competitive position
3. **Growth Analyst** — Evaluates growth metrics and KPIs
4. **Valuation Specialist** — Applies RAG-retrieved valuation parameters
5. **Investment Advisor** — Synthesizes final BUY/HOLD/SELL recommendation

See `AGENTS.md` for detailed agent instructions and `ARCHITECTURE_REVIEW.md` for a full system review.

## Project Layout

```
├── main.py                  # CLI entry point
├── orchestrator.py          # Workflow coordinator
├── agents.py                # LLM agent factory (5 agents)
├── tasks.py                 # Pipeline step definitions
├── vision_extractor.py      # PDF text extraction (PyMuPDF + vision API)
├── valuation_rag.py         # RAG retrieval for valuation parameters
├── utils.py                 # Ollama connectivity helpers
├── create_vectordb.py       # Standalone vector DB builder
├── web/                     # FastAPI web interface
│   ├── main.py
│   └── static/
└── tests/                   # Test suite (pytest)
```
