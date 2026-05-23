"""
main.py — CLI entry point for the Financial Analysis Agentic Application.
"""
import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from orchestrator import FinancialAnalysisOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial Analysis Agentic Application"
    )
    parser.add_argument(
        "--company-name", type=str, default=None,
        help="Company name (optional — auto-extracted from documents if omitted).",
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
    print(f"  Company Name:      {COMPANY_NAME or '(auto-detect)'}")
    print(f"  Output Path:       {OUTPUT_PATH}")
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

        print("STEP 1: Extracting Financial Documents")
        print("-" * 80)
        all_docs = orchestrator.extract_financial_documents()
        print(f"\n✓ {len(all_docs)} document(s) accepted\n")

        def _save_report(report: str, company_name: str, doc_label: str = "") -> str:
            label = f"_{doc_label}" if doc_label else ""
            output_filename = (
                f"investment_report_{company_name.replace(' ', '_')}{label}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            output_file = os.path.join(OUTPUT_PATH, output_filename)
            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write(report)
            print(f"  ✓ Report saved to: {output_file}")
            return output_file

        if len(all_docs) == 1:
            report, company_name = orchestrator.run_analysis(
                COMPANY_NAME, extracted_docs=all_docs
            )
            output_file = _save_report(report, company_name)
            print(f"\n{'='*80}")
            print("ANALYSIS COMPLETE!")
            print(f"{'='*80}")
            print(f"\n✓ Report saved to: {output_file}")
            print(f"\nTo view:  code {output_file}\n")
            print("\n" + "=" * 80)
            print("REPORT PREVIEW")
            print("=" * 80)
            print(report)
        else:
            print("=" * 80)
            print(f"GENERATING PER-FILE REPORTS ({len(all_docs)} files)")
            print("=" * 80)
            for doc_name, doc_content in all_docs.items():
                single_doc = {doc_name: doc_content}
                print(f"\n--- Analyzing: {doc_name} ---")
                report, company_name = orchestrator.run_analysis(
                    COMPANY_NAME, extracted_docs=single_doc
                )
                doc_base = os.path.splitext(doc_name)[0]
                _save_report(report, company_name, doc_label=doc_base)
            print(f"\n{'='*80}")
            print("ALL REPORTS COMPLETE!")
            print(f"{'='*80}")

        print("\nArchiving source PDFs to output folder...")
        for doc_name in all_docs:
            src = os.path.join(FILE_SHARE_PATH, doc_name)
            dst = os.path.join(OUTPUT_PATH, doc_name)
            if os.path.exists(src):
                os.rename(src, dst)
                print(f"  ✓ Moved: {doc_name}")

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
