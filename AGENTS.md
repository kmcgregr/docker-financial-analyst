# AGENTS.md

This file provides the essential commands and conventions an OpenCode agent needs to know to work efficiently in this repository.

## Environment Setup

- **Prerequisites**: Python 3.11+, Ollama (running).
- **Required Ollama models**:
  - Vision model: configured via `VISION_MODEL` (default `qwen2.5vl:7b`).
  - Analysis model: `ANALYSIS_MODEL` (default `gpt-oss:20b`).
  - Embedding model: `EMBEDDING_MODEL` (default `nomic-embed-text`).
- Run `./setup.sh` or `.\setup.ps1` to validate prerequisites, create directories, optionally pull missing models, and install Python deps.

## Running the Analysis

**Standard run** (company name auto-extracted from documents):
```bash
python -m main
```

**Override company name at runtime**:
```bash
python -m main --company-name "Apple Inc"
```

**Run the web interface**:
```bash
uvicorn web.main:app --reload
```

## File Layout & Important Paths

- `data/financials/` – place quarterly/annual report PDFs here.
- `data/valuation_parameters.pdf` – contains valuation guidance.
- `data/output/` – reports are stored here.
- `.env` – configuration. Key variables: `FILE_SHARE_PATH`, `VALUATION_PDF_PATH`, `COMPANY_NAME`, `OUTPUT_PATH`, `OLLAMA_BASE_URL`, `VISION_MODEL`, `ANALYSIS_MODEL`, `EMBEDDING_MODEL`.

## Agent‑Specific Notes

- `agents.py` creates a `FinancialAgents` factory that validates model availability at startup. If a model is missing, the process aborts with an informative error.
- The `FinancialAnalysisOrchestrator` in `main.py` orchestrates the workflow:
  1. Extracts PDFs via `VisionDocumentExtractor`.
  2. Builds task pipelines via `FinancialTasks`.
  3. Runs valuation RAG with a targeted query derived from the growth analyst’s summary.
- `vision_extractor.py` and `valuation_rag.py` are pure Python components.

## Common Commands

- **Pull required models**:
```bash
ollama pull qwen2.5vl:7b
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```
- **Check Ollama**:
```bash
curl http://localhost:11434/api/tags
```
- **Install dependencies**:
```bash
pip install -r requirements.txt
```
- **Run tests**:
```bash
python -m pytest tests/ -v
```
- **Clean workspace**:
```bash
rm -rf data/output/*
```

## Troubleshooting

- **Ollama not running** – start the Ollama service (Windows: system tray; macOS/Linux: `ollama serve`).
- **Missing models** – run `./setup.sh` or manually pull them.
- **Vision extraction failures** – verify that `VISION_MODEL` matches a running model and that the PDF contains text or images compatible with the model.
- **No output reports** – ensure `data/valuation_parameters.pdf` is present and PDFs exist in `data/financials/`.

---

This file is intentionally concise: it lists only information that an agent would otherwise mis‑guess. New contributors should refer to this before making changes or running the system.
