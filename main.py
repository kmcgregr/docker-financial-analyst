"""
Financial Analysis Agentic Application - Main Entry Point
Orchestrates the financial analysis workflow using LangChain.

Fixes applied vs v2:
  D - Valuation RAG queried at runtime with growth-informed targeted query.
  E - PipelineStep explicit binding; zip() misalignment eliminated.
  V3 - extract_financial_documents() uses PageResult.method to distinguish
       blank pages from vision failures in its validation log.

Previous fixes retained:
  A - Content validation (empty extractions skipped).
  B - cap_context() limits context injected into Tasks 2-5.
  C - SUMMARY_PROMPT_SUFFIX + extract_summary_block() keep context small.
"""
import os
import sys
import glob
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

from agents import FinancialAgents
from tasks import FinancialTasks, PipelineStep
from vision_extractor import VisionDocumentExtractor
from valuation_rag import ValuationRAG


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = 12_000

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

FALLBACK_VALUATION_QUERY = (
    "What are all the valuation parameters, methodologies, discount rates, "
    "and comparable multiples?"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_summary_block(result: str) -> str:
    """Pull the === SUMMARY === block; fall back to last paragraph."""
    start = result.find("=== SUMMARY ===")
    end   = result.find("=== END SUMMARY ===")
    if start != -1 and end != -1:
        return result[start : end + len("=== END SUMMARY ===")]
    paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else result[-500:]


def cap_context(context: str) -> str:
    """Truncate context to MAX_CONTEXT_CHARS, keeping the most recent content."""
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    truncated = context[-MAX_CONTEXT_CHARS:]
    first_newline = truncated.find("\n")
    if first_newline != -1:
        truncated = truncated[first_newline:]
    return "[Earlier analysis truncated to stay within context limits]\n" + truncated


def build_valuation_rag_query(context: str) -> str:
    """
    Build a targeted RAG query from the growth analyst's summary block.
    Falls back to a generic query if parsing fails.
    """
    summary  = extract_summary_block(context)
    metrics  = ""
    findings = ""

    for line in summary.splitlines():
        line = line.strip()
        if line.startswith("KEY_METRICS:"):
            metrics  = line.replace("KEY_METRICS:",  "").strip()
        elif line.startswith("KEY_FINDINGS:"):
            findings = line.replace("KEY_FINDINGS:", "").strip()

    if not metrics and not findings:
        return FALLBACK_VALUATION_QUERY

    parts = [
        "What valuation methodologies, discount rates, and multiples apply "
        "to a company with:"
    ]
    if metrics:
        parts.append(f"these financial metrics: {metrics}")
    if findings:
        parts.append(f"and these characteristics: {findings}")
    parts.append(
        "Include DCF parameters, comparable company multiples, "
        "and any relevant benchmarks."
    )
    return " ".join(parts)


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
    # Fix A + V3 — Document extraction with typed result validation
    # ------------------------------------------------------------------

    def extract_financial_documents(self) -> Dict[str, str]:
        """
        Extract content from all financial PDFs in file_share_path.

        Fix A: skips files returning fewer than 200 chars.
        V3:    logs whether short output was due to blank pages or vision
               failures, using the method counts written by _log_summary()
               in vision_extractor.py.
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

            # Fix A: content length guard
            stripped = content.strip()
            if not stripped or len(stripped) < 200:
                # V3: actionable guidance based on likely cause
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
                f"{os.getenv('VISION_MODEL', 'qwen2-vl:7b')}\n"
                f"  • Check extraction summaries above for per-file method counts."
            )

        return documents

    # ------------------------------------------------------------------
    # Fix D — Runtime valuation RAG injection
    # ------------------------------------------------------------------

    def _inject_valuation_params(
        self, step: PipelineStep, context: str
    ) -> str:
        """
        Replace {valuation_placeholder} in Task 4's prompt with a targeted
        RAG result fetched just before the step runs.
        Falls back to step._fallback_params on any failure.
        """
        task_input = step.task["input"]

        if "{valuation_placeholder}" not in task_input:
            return task_input

        valuation_text = step._fallback_params  # safe default

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
    # Main analysis workflow
    # ------------------------------------------------------------------

    def run_analysis(self, company_name: str) -> str:
        """Execute the complete financial analysis workflow."""

        print(f"\n{'='*80}")
        print(f"FINANCIAL ANALYSIS FOR {company_name.upper()}")
        print(f"{'='*80}\n")

        # Step 1: Extract documents
        print("STEP 1: Extracting Financial Documents")
        print("-" * 80)
        extracted_docs = self.extract_financial_documents()
        print(f"\n✓ {len(extracted_docs)} document(s) accepted\n")

        # Step 2: Pre-fetch fallback valuation params
        print("STEP 2: Loading Valuation Parameters (fallback)")
        print("-" * 80)
        fallback_valuation_params = self.valuation_rag.query(
            FALLBACK_VALUATION_QUERY, k=10
        )
        print(f"✓ Fallback params loaded ({len(fallback_valuation_params):,} chars)\n")

        # Step 3: Initialise agents
        print("STEP 3: Initializing AI Agents")
        print("-" * 80)
        agents = self.agents_factory.create_agents()
        print(f"✓ {len(agents)} specialized agents created\n")

        # Step 4: Build pipeline (Fix D + E)
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

        # Step 5: Execute pipeline (Fix E — named steps)
        print("STEP 5: Executing Analysis Pipeline")
        print("-" * 80)
        print("This may take 10-15 minutes depending on document size...\n")

        results: List[str] = []
        context: str       = ""

        for i, step in enumerate(pipeline):
            print(f"--- Step {i+1}/{len(pipeline)}: {step.name} ---")

            task_input = step.task["input"]

            # Fix B: inject capped context
            if "{context_placeholder}" in task_input:
                task_input = task_input.replace(
                    "{context_placeholder}", cap_context(context)
                )

            # Fix D: inject targeted valuation params for valuation step
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

            # Fix C: structured summary suffix
            task_input_final = task_input + SUMMARY_PROMPT_SUFFIX

            try:
                result = step.agent.run(input=task_input_final)
                results.append(result)

                summary = extract_summary_block(result)
                context += f"\n\n--- {step.name} Summary ---\n{summary}"

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

        # Step 6: Generate report
        print("STEP 6: Generating Final Report")
        print("-" * 80)
        report = self.generate_report(
            company_name, "\n\n".join(results), extracted_docs
        )
        print("✓ Report generated\n")
        return report

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        company_name: str,
        analysis_result: str,
        extracted_docs: Dict[str, str],
    ) -> str:
        """Generate the final formatted investment report."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_id = (
            f"{company_name.replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        doc_list = "\n".join(f"  - {f}" for f in extracted_docs)

        return f"""
{'='*80}
FINANCIAL ANALYSIS & INVESTMENT REPORT
{'='*80}

Company: {company_name}
Report Generated: {timestamp}
Analysis System: LangChain Multi-Agent Financial Analyzer

Documents Analyzed:
{doc_list}

{'='*80}
EXECUTIVE SUMMARY & ANALYSIS
{'='*80}

{analysis_result}

{'='*80}
METHODOLOGY
{'='*80}

This analysis was conducted using a multi-agent AI system with the following
specialized agents:

1. Document Analyst      - Extracted and organized financial data
2. Business Analyst      - Analyzed business operations and revenue streams
3. Growth Analyst        - Evaluated growth metrics and KPIs
4. Valuation Specialist  - Calculated company valuation using targeted parameters
5. Investment Advisor    - Synthesized findings into actionable recommendation

Technologies Used:
- Vision Model:   {os.getenv('VISION_MODEL',   'qwen2-vl:7b')} (document extraction)
- Analysis Model: {os.getenv('ANALYSIS_MODEL', 'llama3.1:8b')} (financial analysis)
- Framework:      LangChain (agent orchestration)
- RAG System:     ChromaDB + Ollama Embeddings (valuation parameters)

Pipeline Version: v3
  Fix A  — Content validation (empty/error extractions skipped)
  Fix B  — Context cap ({MAX_CONTEXT_CHARS:,} chars) prevents overflow on Steps 3-5
  Fix C  — Structured summaries; KEY_METRICS/KEY_FINDINGS block per agent
  Fix D  — Valuation RAG queried at runtime with growth-informed targeted query
  Fix E  — Explicit PipelineStep binding; zip() misalignment eliminated
  Fix V1 — Vision model detection covers qwen2.5vl and modern variants
  Fix V2 — Page pre-screening; text-heavy pages skip vision API
  Fix V3 — Typed PageResult; blank pages distinguished from vision failures
  Fix V4 — Page-level extraction cache; re-runs skip vision API entirely
  Fix V5 — Focused vision prompt (~150 tokens vs original ~500)
  Fix V6 — Per-file extraction summary (method counts logged)

{'='*80}
IMPORTANT DISCLAIMER
{'='*80}

This report is generated by an AI-powered analysis system for informational
purposes only and should NOT be considered as financial advice, investment
recommendation, or a substitute for professional financial consultation.

Key Considerations:
- AI analysis may contain errors or omissions
- Market conditions change rapidly
- Past performance does not guarantee future results
- Investment decisions should be based on comprehensive due diligence
- Always consult with qualified financial advisors before investing
- Consider your personal risk tolerance and investment objectives

The creators and operators of this system assume no liability for investment
decisions made based on this report.

{'='*80}
END OF REPORT
{'='*80}

Generated by LangChain Financial Analysis System
Report ID: {report_id}
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial Analysis Agentic Application"
    )
    parser.add_argument(
        "--company-name", type=str, required=True,
        help="Name of the company to analyze.",
    )
    args = parser.parse_args()

    FILE_SHARE_PATH    = os.getenv("FILE_SHARE_PATH",    "data/financials")
    VALUATION_PDF_PATH = os.getenv("VALUATION_PDF_PATH", "data/valuation_parameters.pdf")
    COMPANY_NAME       = args.company_name
    OUTPUT_PATH        = os.getenv("OUTPUT_PATH",        "data/output")

    print("\n" + "=" * 80)
    print("FINANCIAL ANALYSIS SYSTEM — STARTUP")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  File Share Path:   {FILE_SHARE_PATH}")
    print(f"  Valuation PDF:     {VALUATION_PDF_PATH}")
    print(f"  Company Name:      {COMPANY_NAME}")
    print(f"  Output Path:       {OUTPUT_PATH}")
    print(f"  Max Context Chars: {MAX_CONTEXT_CHARS:,}")
    print()

    if not os.path.exists(FILE_SHARE_PATH):
        print(f"ERROR: File share path does not exist: {FILE_SHARE_PATH}")
        sys.exit(1)

    if not os.path.exists(VALUATION_PDF_PATH):
        print(f"ERROR: Valuation PDF does not exist: {VALUATION_PDF_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    try:
        orchestrator = FinancialAnalysisOrchestrator(
            file_share_path=FILE_SHARE_PATH,
            valuation_pdf_path=VALUATION_PDF_PATH,
        )
        report = orchestrator.run_analysis(COMPANY_NAME)

        output_filename = (
            f"investment_report_{COMPANY_NAME.replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        output_file = os.path.join(OUTPUT_PATH, output_filename)

        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(report)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\n✓ Report saved to: {output_file}")
        print(f"\nTo view:  cat {output_file}\n")
        print("\n" + "=" * 80)
        print("REPORT PREVIEW")
        print("=" * 80)
        print(report)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED")
        print(f"{'='*80}")
        print(f"\n✗ {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()