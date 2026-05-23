"""
orchestrator.py
Financial Analysis Orchestrator — coordinates the multi-agent analysis pipeline.

All helpers and FinancialAnalysisOrchestrator live here.
main.py and web/main.py both import from this module.
"""
from __future__ import annotations

import os
import glob
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

from agents import FinancialAgents
from tasks import FinancialTasks, PipelineStep
from utils import (
    MAX_CONTEXT_CHARS,
    FALLBACK_VALUATION_QUERY,
    cap_context,
    build_valuation_rag_query,
)
from vision_extractor import VisionDocumentExtractor
from valuation_rag import ValuationRAG


SUMMARY_PROMPT_SUFFIX = """

---
IMPORTANT: After completing your analysis, output a STRUCTURED SUMMARY BLOCK
in exactly this format — do not skip it:

=== SUMMARY ===
KEY_METRICS: [up to 5 specific numbers/metrics, e.g. "Revenue: $234M, +23% YoY"]
KEY_FINDINGS: [up to 3 one-sentence conclusions]
CONFIDENCE: [HIGH / MEDIUM / LOW — based on data quality available]
=== END SUMMARY ===

This summary will be read by the next analyst in the pipeline.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_company_name_simple(extracted_docs: Dict[str, str],
                                  analysis_llm) -> str:
    """
    Extract company name from the first extracted document.

    Falls back: env var COMPANY_NAME → first PDF filename → "Unknown Company".
    """
    fallback = os.getenv("COMPANY_NAME")
    if fallback:
        return fallback

    if extracted_docs:
        first_filename = next(iter(extracted_docs))
        first_content = list(extracted_docs.values())[0][:3000]
        print("  Extracting company name from documents...")
        try:
            template = (
                "Read this financial document excerpt and identify "
                "the company name.\n"
                "Return ONLY the company name — nothing else.\n\n"
                "Document excerpt:\n{document_excerpt}"
            )
            prompt = PromptTemplate(
                input_variables=["document_excerpt"],
                template=template,
            )
            result = analysis_llm.invoke(
                prompt.format(document_excerpt=first_content)
            )
            name = result.strip().strip('"').strip("'")
            if name:
                print(f"  ✓ Extracted company name: {name}")
                return name
        except Exception as exc:
            print(f"  Could not extract company name via LLM ({exc})")
            pass

        name = os.path.splitext(first_filename)[0].replace("_", " ").replace("-", " ")
        print(f"  Using filename-derived name: {name}")
        return name

    return "Unknown Company"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class FinancialAnalysisOrchestrator:
    """Main orchestrator for the financial analysis workflow."""

    def __init__(self, file_share_path: str, valuation_pdf_path: str) -> None:
        self.file_share_path    = file_share_path
        self.valuation_pdf_path = valuation_pdf_path

        print("Initializing components...")
        self.vision_extractor = VisionDocumentExtractor()
        self.valuation_rag    = ValuationRAG(valuation_pdf_path)
        self.agents_factory   = FinancialAgents()
        self.tasks_factory    = FinancialTasks()
        print("Components initialized successfully\n")

    # ------------------------------------------------------------------
    # Document extraction
    # ------------------------------------------------------------------

    def extract_financial_documents(self) -> Dict[str, str]:
        """
        Extract content from all financial PDFs in file_share_path.
        """
        documents: Dict[str, str] = {}
        pdf_files = glob.glob(os.path.join(self.file_share_path, "*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.file_share_path}")

        print(f"Found {len(pdf_files)} PDF file(s) to process")

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            print(f"\n  Extracting {filename}...")

            try:
                content = self.vision_extractor.extract_from_pdf(pdf_path)
            except FileNotFoundError as exc:
                print(f"  ✗ File not found: {pdf_path} — {exc}")
                continue
            except Exception as exc:
                print(f"  ✗ Extraction error for {filename}: {exc}")
                continue

            stripped = content.strip()
            if not stripped or len(stripped) < 200:
                print(
                    f"  ✗ Skipping {filename}: only {len(stripped)} chars extracted.\n"
                    f"     Possible causes:\n"
                    f"       • Vision model not running  → check: ollama list\n"
                    f"       • Scanned/image-only PDF    → pre-process with OCR\n"
                    f"       • Model detection mismatch  → check VISION_MODEL in .env\n"
                    f"     Extraction method summary is printed above."
                )
                continue

            documents[filename] = content
            print(f"  ✓ {len(content):,} chars accepted from {filename}")

        if not documents:
            raise ValueError(
                "No documents were successfully extracted.\n"
                f"  • Confirm PDF files exist in:       {self.file_share_path}\n"
                f"  • Confirm vision model is running:  "
                f"{os.getenv('VISION_MODEL', 'qwen2.5vl:7b')}\n"
                f"  • Check extraction summaries above for per-file method counts."
            )

        return documents

    # ------------------------------------------------------------------
    # Runtime valuation RAG injection
    # ------------------------------------------------------------------

    def _inject_valuation_params(self, step: PipelineStep, context: str) -> str:
        """
        Replace {valuation_placeholder} in Task 4's prompt with a targeted
        RAG result fetched just before the step runs.
        Falls back to step._fallback_params on any failure.
        """
        task_input = step.task["input"]

        if "{valuation_placeholder}" not in task_input:
            return task_input

        valuation_text = step._fallback_params

        if step._rag_query_fn is not None:
            targeted_query = build_valuation_rag_query(context)
            print(f"  RAG query: {targeted_query[:120]}...")
            try:
                valuation_text = step._rag_query_fn(targeted_query)
                print(
                    f"  ✓ RAG returned {len(valuation_text):,} chars of "
                    "targeted valuation params"
                )
            except Exception as exc:
                print(f"  ✗ RAG query failed ({exc}) — using fallback params")
        else:
            print("  No rag_query_fn attached — using fallback valuation params")

        return task_input.replace("{valuation_placeholder}", valuation_text)

    # ------------------------------------------------------------------
    # Company name extraction
    # ------------------------------------------------------------------

    def _extract_company_name(self, extracted_docs: Dict[str, str]) -> str:
        return _extract_company_name_simple(
            extracted_docs, self.agents_factory.analysis_llm
        )

    # ------------------------------------------------------------------
    # Main analysis workflow
    # ------------------------------------------------------------------

    def run_analysis(
        self,
        company_name: Optional[str] = None,
        extracted_docs: Optional[Dict[str, str]] = None,
    ) -> tuple[str, str]:
        """
        Execute the complete financial analysis workflow.

        Args:
            company_name: Optional company name. If None, extracted from documents.
            extracted_docs: Optional pre-extracted documents. If None, extracted
                            from PDFs in file_share_path.

        Returns:
            Tuple of (report_text, company_name).
        """

        if extracted_docs is None:
            print("STEP 1: Extracting Financial Documents")
            print("-" * 80)
            extracted_docs = self.extract_financial_documents()
            print(f"\n✓ {len(extracted_docs)} document(s) accepted\n")

        if company_name is None:
            company_name = self._extract_company_name(extracted_docs)

        print(f"\n{'='*80}")
        print(f"FINANCIAL ANALYSIS FOR {company_name.upper()}")
        print(f"{'='*80}\n")

        print("STEP 2: Loading Valuation Parameters (fallback)")
        print("-" * 80)
        fallback_valuation_params = self.valuation_rag.query(
            FALLBACK_VALUATION_QUERY, k=10
        )
        print(f"✓ Fallback params loaded ({len(fallback_valuation_params):,} chars)\n")

        print("STEP 3: Initializing AI Agents")
        print("-" * 80)
        agents = self.agents_factory.create_agents()
        print(f"✓ {len(agents)} specialized agents created\n")

        print("STEP 4: Building Analysis Pipeline")
        print("-" * 80)
        pipeline: List[PipelineStep] = self.tasks_factory.create_pipeline(
            agents=agents,
            extracted_docs=extracted_docs,
            valuation_params=fallback_valuation_params,
            company_name=company_name,
            rag_query_fn=lambda q: self.valuation_rag.query(q, k=6),
        )
        print(f"✓ {len(pipeline)} pipeline steps bound:")
        for i, step in enumerate(pipeline, 1):
            print(f"  {i}. {step.name}")
        print()

        print("STEP 5: Executing Analysis Pipeline")
        print("-" * 80)
        print("This may take 10-15 minutes depending on document size...\n")

        results: List[str] = []
        # Seed context with extracted documents so all agents have raw data
        all_docs_text = "\n\n".join(
            f"--- {fname} ---\n{content}" for fname, content in extracted_docs.items()
        )
        context = cap_context(all_docs_text)

        for i, step in enumerate(pipeline):
            print(f"--- Step {i+1}/{len(pipeline)}: {step.name} ---")

            task_input = step.task["input"]

            if "{context_placeholder}" in task_input:
                task_input = task_input.replace(
                    "{context_placeholder}", cap_context(context)
                )

            if step.name == "Valuation Specialist":
                task_input = self._inject_valuation_params(
                    PipelineStep(
                        name=step.name,
                        agent=step.agent,
                        task={"input": task_input},
                        _rag_query_fn=step._rag_query_fn,
                        _fallback_params=step._fallback_params,
                    ),
                    context,
                )

            task_input_final = task_input + SUMMARY_PROMPT_SUFFIX

            try:
                result = step.agent.chain.invoke({"input": task_input_final})
                results.append(result)

                context += f"\n\n--- {step.name} Output ---\n{result}"
                context = cap_context(context)

                print(f"✓ {step.name} completed ({len(result):,} chars)")
                print(f"  Accumulated context: {len(context):,} chars\n")

            except Exception as exc:
                error_msg = f"{step.name} encountered an error: {exc}"
                print(f"✗ {error_msg}")
                results.append(error_msg)
                context += (
                    f"\n\n--- {step.name} Summary ---\n"
                    f"KEY_FINDINGS: This step failed — {exc}\n"
                    f"CONFIDENCE: LOW"
                )
                print(f"--- {step.name} failed, continuing ---\n")

        print("\n✓ Analysis pipeline completed\n")

        print("STEP 6: Generating Final Report")
        print("-" * 80)
        report = self.generate_report(
            company_name, "\n\n".join(results), extracted_docs
        )
        print("✓ Report generated\n")
        return report, company_name

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        company_name: str,
        analysis_result: str,
        extracted_docs: Dict[str, str],
    ) -> str:
        """Generate the final formatted investment report (Markdown)."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_id = (
            f"{company_name.replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        doc_list = "\n".join(f"  - {f}" for f in extracted_docs)

        return f"""
# Financial Analysis & Investment Report

**Company:** {company_name}  
**Report Generated:** {timestamp}  
**Analysis System:** LangChain Multi-Agent Financial Analyzer

---

## Documents Analyzed

{doc_list}

---

## Executive Summary & Analysis

{analysis_result}

---

## Methodology

This analysis was conducted using a multi-agent AI system with the following
specialized agents:

1. **Document Analyst** — Extracted and organized financial data
2. **Business Analyst** — Analyzed business operations and revenue streams
3. **Growth Analyst** — Evaluated growth metrics and KPIs
4. **Valuation Specialist** — Calculated company valuation using targeted parameters
5. **Investment Advisor** — Synthesized findings into actionable recommendation

### Technologies Used

- **Vision Model:** {os.getenv('VISION_MODEL',   'qwen2-vl:7b')} (document extraction)
- **Analysis Model:** {os.getenv('ANALYSIS_MODEL', 'llama3.1:8b')} (financial analysis)
- **Framework:** LangChain (agent orchestration)
- **RAG System:** ChromaDB + Ollama Embeddings (valuation parameters)

### Pipeline Version: v3

- Fix A — Content validation (empty/error extractions skipped)
- Fix B — Context cap ({MAX_CONTEXT_CHARS:,} chars) prevents overflow on Steps 3-5
- Fix C — Structured summaries; KEY_METRICS/KEY_FINDINGS block per agent
- Fix D — Valuation RAG queried at runtime with growth-informed targeted query
- Fix E — Explicit PipelineStep binding; zip() misalignment eliminated
- Fix V1 — Vision model detection covers qwen2.5vl and modern variants
- Fix V2 — Page pre-screening; text-heavy pages skip vision API
- Fix V3 — Typed PageResult; blank pages distinguished from vision failures
- Fix V4 — Page-level extraction cache; re-runs skip vision API entirely
- Fix V5 — Focused vision prompt (~150 tokens vs original ~500)
- Fix V6 — Per-file extraction summary (method counts logged)

---

## Important Disclaimer

This report is generated by an AI-powered analysis system for informational
purposes only and should **NOT** be considered as financial advice, investment
recommendation, or a substitute for professional financial consultation.

### Key Considerations

- AI analysis may contain errors or omissions
- Market conditions change rapidly
- Past performance does not guarantee future results
- Investment decisions should be based on comprehensive due diligence
- Always consult with qualified financial advisors before investing
- Consider your personal risk tolerance and investment objectives

The creators and operators of this system assume no liability for investment
decisions made based on this report.

---

*Generated by LangChain Financial Analysis System*  
*Report ID: {report_id}*
"""
