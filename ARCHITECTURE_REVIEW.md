# Architecture Review: Financial Analyst

> **Note:** Since this review was written, `Dockerfile`, `web/Dockerfile`, and
> `docker-compose.yml` have been removed. The project now runs directly in a
> Python virtual environment. All Docker-specific elements in this document
> are historical.

**Review Date:** 2026-05-16
**Codebase:** Multi-agent AI system that analyzes financial PDF documents and generates investment reports using local LLMs via Ollama.
**Status:** Functional with known issues

---

## 1. Executive Summary

This codebase implements a five-agent sequential pipeline that extracts financial data from PDFs, analyzes business quality, growth, and valuation, then produces a structured investment report. The architecture is well-modularized with clean separations between extraction, analysis, RAG retrieval, and orchestration. However, it has one critical blocker — the `orchestrator.py` module is an empty directory, with the actual code living in a misnamed `orchestrator,py` file (comma instead of dot). Additionally, all three prior run outputs show every agent stopping at iteration/time limits, meaning the pipeline has never produced a real analysis. The most important action is to fix the filename and switch agents to a non-iterative chain to stop the iteration-limit failures.

---

## 2. Component Inventory

| Module | Role | Pattern |
|--------|------|---------|
| `main.py` | CLI entry point; parses args, validates paths, starts orchestrator | Bootstrap |
| `orchestrator,py` | Central workflow coordinator | Orchestrator / Sequential Pipeline |
| `agents.py` | Creates 5 analysis agents (LLMChains) | Factory |
| `tasks.py` | Defines prompts and builds PipelineStep list | Factory + Declarative Pipeline |
| `vision_extractor.py` | Extracts text from PDF pages via PyMuPDF or vision API | Strategy + Cache-aside |
| `valuation_rag.py` | RAG retrieval for valuation parameters from a reference PDF | RAG (Retrieval-Augmented Generation) |
| `utils.py` | Ollama connectivity check with exception-based error handling | Utility |
| `create_vectordb.py` | Standalone tool to build persistent Chroma vector stores from PDFs | Utility |
| `web/main.py` | FastAPI web server; file upload and on-demand analysis | Web API |
| `web/static/index.html` | Drag-and-drop upload UI | Single-page frontend |
| `Dockerfile` | Container build for batch analysis | Docker |
| `web/Dockerfile` | Container build for web service | Docker |
| `docker-compose.yml` | Multi-service orchestration (app + web) | Docker Compose |
| `setup.sh` / `setup.ps1` | Environment validation and setup | Script |
| `AGENTS.md` | OpenCode agent instructions | Documentation |

---

## 3. Architectural Flow

A single representative run traces through these stages:

1. **Entry** — User runs `docker-compose up` or `python -m main --company-name "X"`
2. **Bootstrap** — `main.py` parses `--company-name`, reads env vars, validates input paths, creates `FinancialAnalysisOrchestrator`
3. **Init** — Orchestrator `__init__` creates `VisionDocumentExtractor`, `ValuationRAG`, `FinancialAgents`, `FinancialTasks`
4. **Extraction** — Orchestrator calls `extract_financial_documents()`; `VisionDocumentExtractor.extract_from_multiple_pdfs()` reads each PDF page-by-page, using PyMuPDF text where sufficient (>100 chars) and vision API for sparse pages; results cached to `.extraction_cache/`
5. **RAG pre-fetch** — Orchestrator does a broad fallback query on `ValuationRAG` (k=10) for use as fallback if the later targeted query fails
6. **Agent creation** — `FinancialAgents.create_agents()` builds 5 `LLMChain` instances, each with a role-specific prompt template
7. **Pipeline build** — `FinancialTasks.create_pipeline()` binds agents to tasks via `PipelineStep` objects by role name; attaches `_rag_query_fn` to the Valuation step
8. **Pipeline execution** — Sequential loop over 5 steps:
    - **Step 1** (Document Analyst): Receives raw PDF text → produces structured financial data extraction
    - **Step 2** (Business Analyst): Receives Step 1's summary → analyzes business model, revenue streams, competitive moat
    - **Step 3** (Growth Analyst): Receives cumulative context → calculates growth rates, KPIs, pricing power
    - **Step 4** (Valuation Specialist): Prompt's `{valuation_placeholder}` is replaced at runtime via `_inject_valuation_params()`; uses a targeted RAG query built from Step 3's structured summary (e.g. "40% gross margins, SaaS model") → produces valuation
    - **Step 5** (Investment Advisor): Receives all prior context → synthesizes final BUY/HOLD/SELL recommendation
9. **Report** — Orchestrator assembles results into a formatted report and writes to `data/output/`

```
main.py (CLI)
   │
   └─── FinancialAnalysisOrchestrator (orchestrator,py)
            │
            ├─── VisionDocumentExtractor ─── PyMuPDF + Ollama Vision → PDF text
            │         └── .extraction_cache/ (JSON per PDF)
            │
            ├─── ValuationRAG ─── PyMuPDF → Chroma (in-memory) → semantic search
            │         └── OllamaEmbeddings (nomic-embed-text)
            │
            ├─── FinancialAgents (factory) → 5 × LLMChain
            │         └── Ollama (analysis model)
            │
            └─── FinancialTasks (factory) → PipelineStep[5]
                      │
                      ├── Step 1: Document Analyst
                      ├── Step 2: Business Analyst
                      ├── Step 3: Growth Analyst
                      ├── Step 4: Valuation Specialist ← RAG query at runtime
                      └── Step 5: Investment Advisor → Report

web/main.py (FastAPI)
   │
   └─── same FinancialAnalysisOrchestrator (shared via volume mount)
```

---

## 4. Strengths

### 4.1 Clean modular separation
Each component has a single responsibility: `vision_extractor.py` only extracts PDFs, `valuation_rag.py` only retrieves valuation context, `agents.py` only builds LLM chains, `tasks.py` only defines prompts and binds them to agents. The orchestrator is the single point of coordination. This makes individual components testable and replaceable.

### 4.2 Smart vision extraction with caching
The `VisionDocumentExtractor` (vision_extractor.py:134) uses a tiered approach: PyMuPDF text extraction first, with a 100-character threshold (V2) that skips the vision API for ~80% of text-heavy pages (narrative MD&A, footnotes). Results are typed via the `PageResult` dataclass (V3) with four methods (`text`/`vision`/`fallback`/`empty`), and the full page-level cache (V4) keyed on MD5 of `(path + mtime + model)` means re-runs incur zero API calls. The V6 extraction summary per file lets operators distinguish blank pages from vision failures at a glance.

### 4.3 Explicit pipeline binding over implicit zip
The `PipelineStep` dataclass (tasks.py:32) with role-name-based agent-to-task binding (Fix E) eliminates the classic `zip(agents, tasks)` bug where mismatched list lengths silently drop or misalign steps. The `create_pipeline()` method validates exactly 5 agents with required role names, and raises immediately on mismatch.

### 4.4 Runtime-directed RAG injection
The valuation step's parameters are not determined at pipeline-build time. Instead, the orchestrator extracts a structured `=== SUMMARY ===` block from the growth analyst's output (containing `KEY_METRICS` and `KEY_FINDINGS`), builds a targeted RAG query from that, and injects the result at runtime (Fix D, orchestrator.py:195). This means the valuation specialist receives contextually relevant valuation parameters (e.g. SaaS multiples for a subscription business) rather than a generic dump.

### 4.5 Per-step error resilience
The pipeline execution loop (orchestrator.py:273) wraps each agent call in try/except. If any single agent fails, the error is recorded, a structured failure summary is appended to the context, and execution continues to the next agent. A single model timeout or hallucination does not kill the entire run.

### 4.6 Cross-platform setup
Both `setup.sh` (bash) and `setup.ps1` (PowerShell) scripts validate prerequisites (Docker, Ollama), check model availability with an option to pull missing models, create directory structure, build the Docker image, and offer to start the analysis. This lowers the bar for users on macOS, Linux, and Windows.

---

## 5. Issues and Risks

### 5.1 Critical — `orchestrator.py` is an empty directory, actual code misnamed

**Root cause:** `orchestrator.py` is an empty directory (0 entries). The `FinancialAnalysisOrchestrator` class and all helper functions live in `orchestrator,py` — a filename with a comma instead of a dot before `py`. Python's import system treats `orchestrator` as a namespace package (directory) and finds no `__init__.py` or matching module, so `from orchestrator import FinancialAnalysisOrchestrator` will **always fail with `ModuleNotFoundError`**.

**Consequence:** Both `main.py` and `web/main.py` import `from orchestrator import FinancialAnalysisOrchestrator`. Neither entry point can start — the process crashes on import before the first log line. The docker-compose volume mount `./orchestrator.py:/app/orchestrator.py` would mount an empty directory into the container, not a Python file.

**Evidence:** `orchestrator.py` shows `<type>directory</type><entries></entries>` while `Get-ChildItem` lists both `orchestrator.py` (dir) and `orchestrator,py` (file with commas).

**Fix:** Delete the empty `orchestrator.py` directory and rename `orchestrator,py` → `orchestrator.py`.

### 5.2 High — All agents hit iteration/time limits in every prior run

**Root cause:** The agents are created via `_create_simple_agent()` (agents.py:79) which returns a raw `LLMChain` with `verbose=True`. LangChain's default `LLMChain.run()` wraps the chain in an AgentExecutor with a default `max_iterations` of 15 (or `max_execution_time`). The verbose prompt templates instruct the model to "be thorough and detailed" and produce structured output — the agent repeatedly calls the LLM, generating more and more verbose output, until the iteration limit kills it.

**Consequence:** Every prior output file shows "Agent stopped due to iteration limit or time limit" for all 5 agents. The reports contain no actual financial analysis. The feature is effectively broken.

**Evidence:**
- `investment_report_Shopify_20251123_221656.txt`: Lines 17-24, all 5 agents stopped at iteration limit.
- `investment_report_Coveo_20251125_022052.txt`: Lines 17-25, all 5 agents stopped.
- `investment_report_Shopify_20251123_200148.txt`: Lines 17-25, all 5 "encountered an error".

**Fix:** Replace `LLMChain` with `LLMChain` configured with a simple `PromptTemplate` + `llm` call, not the agent loop. Use `llm.invoke(prompt)` directly instead of `chain.run()`, or set `max_iterations=1` on the underlying agent. The verbose prompts are already instructing the model directly — they do not need ReAct-style tool use, so the agent loop is entirely unnecessary.

### 5.3 Medium — No test coverage

**Root cause:** There are zero test files in the repository. No `tests/` directory, no `pytest` configuration, no `unittest` modules.

**Consequence:** Every change is manually verified by running the full pipeline (10-15 minutes). Regression bugs — especially from the various "fixes" documented in docstrings — have no safety net. The comma-in-filename bug (#5.1) would have been caught by a single import test.

### 5.4 Medium — Setup scripts reference outdated model names

**Root cause:** `setup.sh` and `setup.ps1` hardcode `qwen2-vl:7b` and `llama3.1:8b` as required models, while `.env` uses `qwen2.5vl:7b` and `gpt-oss:20b`. The `setup.sh` health check also references the old `financial_analysis.py` file and `finance-llama-8b`.

**Consequence:** Users running `setup.sh` will be prompted to pull models that don't match what `.env` actually configures. They may pull unnecessary models or miss required ones. The script's `ANALYSIS_MODEL` default (`llama3.1:8b`) differs from `.env`'s `gpt-oss:20b`.

### 5.5 Medium — RAG URL mismatch in default config

**Root cause:** `valuation_rag.py:43` defaults to `http://host.docker.internal:11434` for `OLLAMA_BASE_URL`, while `.env` sets `http://localhost:11434`. The docker-compose overrides this per-service, but anyone running `python -m main` outside Docker would hit the wrong URL and get connection errors from valuation_rag.py (while agents.py and vision_extractor.py use `localhost` from .env).

**Consequence:** Running the orchestrator directly (outside Docker) causes the RAG system to fail connecting to Ollama while the other components work fine, producing a confusing partial-failure mode.

### 5.6 Low — Empty documentation files

**Root cause:** `README.md` and `PROJECT_STRUCTURE.md` are both empty files (0 bytes).

**Consequence:** Users have no authoritative landing documentation. `QUICKSTART.md` fills the gap partially but references old file names and an obsolete project structure.

### 5.7 Medium — Output reports lack actual analysis content

**Root cause:** Beyond the iteration-limit issue (#5.2), the output reports (#5.2 evidence) show that even when the pipeline ran (Shopify older run), the agents returned nonsensical error messages like "PDF file not found: Path to the 10-Q-s-1.pdf document (assuming it's in a directory or folder)" — the model is interpreting the *prompt's description* of what a document is as the error message itself. This suggests the model was not given actual PDF content, or the content passed was insufficient.

**Consequence:** The system has never produced a useful investment report. All three output files are essentially empty.

---

## 6. Dependency Analysis

```
main.py
   └── orchestrator (BROKEN — directory, not file)
         ├── agents.py
         │     └── utils.py
         ├── tasks.py
         ├── vision_extractor.py
         │     ├── utils.py
         │     └── requests (external)
         └── valuation_rag.py
               ├── langchain_community.embeddings.OllamaEmbeddings
               ├── langchain_community.vectorstores.Chroma
               └── langchain_text_splitters

web/main.py
   └── orchestrator (BROKEN — same reason)
         └── [all the same deps as above]

create_vectordb.py  (standalone — no project-internal imports)
   └── fitz, langchain_community, langchain_text_splitters, dotenv

utils.py  (leaf module — no project-internal imports)
```

**Fan-in:**
- `utils.py` is imported by `agents.py` and `vision_extractor.py` — low fan-in, low risk.
- `orchestrator,py` is imported by `main.py` and `web/main.py` — would be high fan-in if the import worked.

**Circular dependencies:** None detected.

**Standalone/modular components:**
- `valuation_rag.py` has no imports from other project files, making it independently testable and replaceable.
- `create_vectordb.py` is fully standalone.
- `vision_extractor.py` imports only `utils.py` from the project.

---

## 7. Recommended Improvements

### Immediate (fix before next run or deploy)

**7.1 Fix the orchestrator filename (resolves #5.1)**

Delete the empty `orchestrator.py` directory and rename `orchestrator,py` to `orchestrator.py`:

```bash
rm -r orchestrator.py
mv orchestrator,py orchestrator.py
```

**7.2 Stop using the LangChain Agent loop (resolves #5.2)**

Replace the `LLMChain` agent pattern with a direct `llm.invoke()` call. The agent loop is unnecessary because no agent uses tools. In `agents.py`, change the agent creation:

Current (`_create_simple_agent`):
```python
chain = LLMChain(llm=self.analysis_llm, prompt=prompt, verbose=True)
chain.role = role
return chain
```

Recommended (replacement in `orchestrator,py` run loop and `agents.py`):
- Add a nonce/`max_iterations=1` parameter, or better yet, just call `llm.invoke()` with the prompt directly in the orchestrator rather than going through the agent loop.
- The `PipelineStep.agent.run()` calls in `orchestrator,py:298` should become `step.agent.invoke(task_input_final)` or simply `self.analysis_llm.invoke(task_input_final)`.

### Medium-term (next sprint or refactor cycle)

**7.3 Add unit tests (resolves #5.3)**

At minimum:
- Import tests verifying every module loads without error.
- A test for `VisionDocumentExtractor` with a synthetic PDF (using PyMuPDF to create an in-memory PDF).
- A test for `ValuationRAG` with a short text file.
- A test for the pipeline binding logic in `tasks.py` verifying role-name validation.

**7.4 Align setup scripts with .env defaults (resolves #5.4)**

Update `setup.sh` and `setup.ps1` to reference `qwen2.5vl:7b` as the default vision model and use the same analysis model default as `.env`. Remove references to `financial_analysis.py` and `finance-llama-8b`.

**7.5 Normalize Ollama URL defaults (resolves #5.5)**

Make `valuation_rag.py` use the same default as the other modules (`localhost:11434`) or derive from a single source of truth. Consider extracting the URL default into `utils.py`.

### Longer-term (architectural evolution)

**7.6 Replace LLMChain with LangGraph or direct invoke**

The entire codebase pins `langchain==0.1.20` and `langchain-community==0.0.38`, which are several major versions behind. `LLMChain` is deprecated in newer LangChain releases. Migrating to `langchain.invoke()` or LangGraph would give better control over execution flow, streaming, and error handling.

**7.7 Make the pipeline truly composable**

The current pipeline is hardcoded to 5 agents in a fixed order. A longer-term improvement would define a pipeline configuration (YAML or dict) specifying agent order, inputs, and enable/disables, so users could run subsets (e.g. "valuation only" or "growth analysis only").

**7.8 Add observability (metrics, tracing, structured logging)**

Currently all output goes to `print()` statements. Adding structured logging (e.g. `loguru` or `structlog`) with levels, plus basic timing metrics per step, would make debugging the 10-15 minute runs substantially easier. LangChain's built-in callbacks could also be used.

---

## 8. Summary Assessment

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Modularity | Good | Clean separation per component; orchestrator is the sole coordinator |
| Reliability | Poor | Critical import bug (#5.1) + all prior runs failed with iteration limits (#5.2); no tests (#5.3) |
| Scalability | Fair | Sequential pipeline scales linearly; Docker compose supports scaling but one agent runs at a time |
| Maintainability | Fair | Well-documented code with fix annotations; but empty README, outdated setup scripts, misnamed file |
| Configuration | Good | Single `.env` file with sensible defaults; docker-compose overrides work correctly |
| Error Handling | Good | Per-step try/except; typed exceptions in utils (U1); fallback for RAG failures |
| Test Coverage | None | Zero test files exist |
| Security | N/A | All-local system (Ollama, no external network); no credentials exposed beyond dummy `OPENAI_API_KEY=NA` |

---

## Key Findings Summary

| # | Severity | Issue | File(s) |
|---|----------|-------|---------|
| 5.1 | Critical | `orchestrator.py` is an empty directory; real code is `orchestrator,py` (comma) | `orchestrator.py/` (dir) vs `orchestrator,py` (file) |
| 5.2 | High | All 5 agents hit iteration/time limits — no analysis ever completes | `agents.py:79-116`, `orchestrator,py:298` |
| 5.3 | Medium | Zero test coverage | — |
| 5.4 | Medium | Setup scripts reference wrong model names | `setup.sh:60`, `setup.ps1:81` |
| 5.5 | Medium | RAG default URL differs from other components | `valuation_rag.py:43` |
| 5.6 | Low | Empty README.md and PROJECT_STRUCTURE.md | `README.md`, `PROJECT_STRUCTURE.md` |
| 5.7 | Medium | All prior output files contain no actual analysis | `data/output/*.txt` |
