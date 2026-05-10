# AGENTS.md

This file provides the essential commands and conventions an OpenCode agent needs to know to work efficiently in this repository.

## Environment Setup

- **Prerequisites**: Docker, Docker‑Compose, Ollama (running).
- **Required Ollama models**:
  - Vision model: configured via `VISION_MODEL` (default `qwen2.5vl:7b`).
  - Analysis model: `ANALYSIS_MODEL` (default `gemma3:12b`).
  - Embedding model: `EMBEDDING_MODEL` (default `nomic-embed-text`).
- Run `./setup.sh` to validate prerequisites, create directories, and optionally pull missing models. It also builds the Docker image.

## Running the Analysis

**Standard run** (use configured company name):
```bash
docker-compose up
```

**Override company name at runtime**:
```bash
COMPANY_NAME="Apple Inc" docker-compose up
```

**Run the Python orchestrator directly** (for debugging or non‑container execution):
```bash
python -m main
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
- `vision_extractor.py` and `valuation_rag.py` are pure Python components; no container‑specific code.

## Common Commands

- **Pull required models**:
```bash
ollama pull qwen2.5vl:7b
ollama pull gemma3:12b
ollama pull nomic-embed-text
```
- **Check Ollama**:
```bash
curl http://localhost:11434/api/tags
```
- **Docker image build**:
```bash
docker-compose build
```
- **Inspect logs**:
```bash
docker-compose logs -f
```
- **Clean workspace**:
```bash
docker-compose down -vm -rf data/output/*
```

## Troubleshooting

- **Ollama not running** – start the Ollama service (Windows: system tray; macOS/Linux: `ollama serve`).
- **Missing models** – run `./setup.sh` or manually pull them.
- **Vision extraction failures** – verify that `VISION_MODEL` matches a running model and that the PDF contains text or images compatible with the model.
- **No output reports** – ensure `COMPANY_NAME` matches a key in at least one PDF and that `data/valuation_parameters.pdf` is present.

---

This file is intentionally concise: it lists only information that an agent would otherwise mis‑guess. New contributors should refer to this before making changes or running the system.
