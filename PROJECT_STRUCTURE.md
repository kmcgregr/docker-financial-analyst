# Project Structure

```
docker-financial-analyst/
│
├── main.py                  # CLI entry point — parses args, runs orchestrator
├── orchestrator.py          # FinancialAnalysisOrchestrator — workflow coordinator
├── agents.py                # FinancialAgents — factory for 5 LLMChain agents
├── tasks.py                 # FinancialTasks — PipelineStep definitions with runtime RAG injection
├── vision_extractor.py      # VisionDocumentExtractor — PDF text via PyMuPDF + vision API fallback
├── valuation_rag.py         # ValuationRAG — Chroma vector store for valuation parameter retrieval
├── utils.py                 # check_model_availability() with typed exceptions
├── create_vectordb.py       # Standalone Chroma vector DB builder from PDFs
│
├── web/                     # FastAPI web interface
│   ├── main.py              # API routes (/, /health, /analyze)
│   └── static/
│       └── index.html       # Drag-and-drop upload UI
│
├── data/
│   ├── financials/          # Place quarterly/annual PDFs here
│   ├── valuation_parameters.pdf  # Reference PDF for valuation guidance
│   └── output/              # Generated investment reports
│
├── tests/                   # pytest test suite
│   ├── __init__.py
│   ├── test_imports.py
│   ├── test_orchestrator_helpers.py
│   └── test_pipeline_binding.py
│
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Build config + pytest settings
│
├── setup.sh                 # Linux/macOS environment setup
├── setup.ps1                # Windows environment setup
│
├── .env                     # Configuration (models, paths, Ollama URL)
├── AGENTS.md                # OpenCode agent instructions
├── ARCHITECTURE_REVIEW.md   # Full system architecture review
└── README.md                # Project overview
```
