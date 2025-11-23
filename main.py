"""
Financial Analysis Agentic Application - Main Entry Point
Orchestrates the financial analysis workflow using LangChain
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import Tool
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

from agents import FinancialAgents
from tasks import FinancialTasks
from vision_extractor import VisionDocumentExtractor
from valuation_rag import ValuationRAG


class FinancialAnalysisOrchestrator:
    """Main orchestrator for the financial analysis workflow"""

    def __init__(self, file_share_path: str, valuation_pdf_path: str):
        self.file_share_path = file_share_path
        self.valuation_pdf_path = valuation_pdf_path

        # Initialize components
        print("Initializing components...")
        self.vision_extractor = VisionDocumentExtractor()
        self.valuation_rag = ValuationRAG(valuation_pdf_path)
        self.agents_factory = FinancialAgents()
        self.tasks_factory = FinancialTasks()
        
        # Create tools
        self.tools = self._create_tools()

        print("Components initialized successfully")

    def _create_tools(self) -> List[Tool]:
        """Create a list of tools for the agents"""
        
        tools = [
            Tool(
                name="Financial Document Extractor",
                func=self.vision_extractor.extract_from_pdf,
                description="Extracts text content from a PDF file.",
            ),
            Tool(
                name="Valuation Parameters Retriever",
                func=self.valuation_rag.query,
                description="Retrieves valuation parameters, methodologies, and formulas.",
            ),
        ]
        return tools

    def extract_financial_documents(self) -> dict:
        """Extract content from all financial documents in file share"""
        import glob

        documents = {}
        pdf_files = glob.glob(os.path.join(self.file_share_path, "*.pdf"))

        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.file_share_path}")

        print(f"\nFound {len(pdf_files)} PDF files to process")

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            print(f"Extracting {filename}...")

            try:
                content = self.vision_extractor.extract_from_pdf(pdf_path)
                documents[filename] = content
                print(f"  ✓ Extracted {len(content)} characters from {filename}")
            except Exception as e:
                print(f"  ✗ Error extracting {filename}: {e}")
                continue

        return documents

    def run_analysis(self, company_name: str) -> str:
        """Execute the complete financial analysis workflow"""

        print(f"\n{'='*80}")
        print(f"FINANCIAL ANALYSIS FOR {company_name.upper()}")
        print(f"{'='*80}\n")

        # Step 1: Extract documents
        print("STEP 1: Extracting Financial Documents")
        print("-" * 80)
        extracted_docs = self.extract_financial_documents()

        if not extracted_docs:
            raise ValueError("No documents were successfully extracted")

        print(f"\n✓ Successfully extracted {len(extracted_docs)} documents\n")

        # Step 2: Query valuation parameters
        print("STEP 2: Loading Valuation Parameters")
        print("-" * 80)
        valuation_params = self.valuation_rag.query(
            "What are all the valuation parameters, methodologies, and formulas?",
            k=10
        )
        print(f"✓ Loaded valuation parameters ({len(valuation_params)} characters)\n")

        # Step 3: Initialize agents
        print("STEP 3: Initializing AI Agents")
        print("-" * 80)
        agents = self.agents_factory.create_agents()
        print(f"✓ Created {len(agents)} specialized agents:")
        # for i, agent in enumerate(agents, 1):
        #     print(f"  {i}. {agent.role}")
        print()

        # Step 4: Create tasks
        print("STEP 4: Creating Analysis Tasks")
        print("-" * 80)
        tasks = self.tasks_factory.create_tasks(
            extracted_docs=extracted_docs,
            valuation_params=valuation_params,
            company_name=company_name
        )
        print(f"✓ Created {len(tasks)} analysis tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task['input'][:60]}...")
        print()

        # Step 5: Execute LangChain workflow
        print("STEP 5: Executing Analysis Workflow")
        print("-" * 80)
        print("This may take 10-15 minutes depending on document complexity...\n")

        # Execute the workflow sequentially
        results = []
        context = ""
        for i, (agent, task) in enumerate(zip(agents, tasks)):
            print(f"--- Running Task {i+1} ---")
            
            # Replace placeholder with the actual context
            if "{context_placeholder}" in task["input"]:
                task["input"] = task["input"].replace("{context_placeholder}", context)
            
            # Invoke the agent
            result = agent.invoke(task)
            
            # Append the result to the list of results
            results.append(result['output'])
            
            # Update the context for the next agent
            context += f"\n\n--- Analysis from previous step ---\n{result['output']}"
            
            print(f"--- Task {i+1} Completed ---")

        final_result = "\n\n".join(results)
        print("\n✓ Analysis workflow completed successfully\n")

        # Step 6: Generate final report
        print("STEP 6: Generating Final Report")
        print("-" * 80)
        report = self.generate_report(company_name, final_result, extracted_docs)
        print("✓ Report generated successfully\n")

        return report

    def generate_report(self, company_name: str, analysis_result: str, extracted_docs: dict) -> str:
        """Generate the final formatted investment report"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc_list = "\n".join([f"  - {filename}" for filename in extracted_docs.keys()])

        report = f"""
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

1. Document Analyst - Extracted and organized financial data
2. Business Model Analyst - Analyzed business operations and revenue streams
3. Growth & Revenue Analyst - Evaluated growth metrics and KPIs
4. Valuation Specialist - Calculated company valuation using multiple methods
5. Investment Advisor - Synthesized findings into actionable recommendation

Technologies Used:
- Vision Model: Qwen2-VL (document extraction)
- Analysis Model: Finance-Llama-8B / Llama3.1 (financial analysis)
- Framework: LangChain (agent orchestration)
- RAG System: ChromaDB + Ollama Embeddings (valuation parameters)

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
Report ID: {company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}
"""
        return report


import argparse


def main():
    """Main execution function"""

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Financial Analysis Agentic Application")
    parser.add_argument("--company-name", type=str, required=True, help="The name of the company to be analyzed.")
    args = parser.parse_args()

    # Load configuration from environment
    FILE_SHARE_PATH = os.getenv('FILE_SHARE_PATH', 'data/financials')
    VALUATION_PDF_PATH = os.getenv('VALUATION_PDF_PATH', 'data/valuation_parameters.pdf')
    COMPANY_NAME = args.company_name
    OUTPUT_PATH = os.getenv('OUTPUT_PATH', 'data/output')

    print("\n" + "="*80)
    print("FINANCIAL ANALYSIS SYSTEM - STARTUP")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  File Share Path: {FILE_SHARE_PATH}")
    print(f"  Valuation PDF: {VALUATION_PDF_PATH}")
    print(f"  Company Name: {COMPANY_NAME}")
    print(f"  Output Path: {OUTPUT_PATH}")
    print()

    # Validate paths
    if not os.path.exists(FILE_SHARE_PATH):
        print(f"ERROR: File share path does not exist: {FILE_SHARE_PATH}")
        sys.exit(1)

    if not os.path.exists(VALUATION_PDF_PATH):
        print(f"ERROR: Valuation PDF does not exist: {VALUATION_PDF_PATH}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    try:
        # Initialize orchestrator
        orchestrator = FinancialAnalysisOrchestrator(
            file_share_path=FILE_SHARE_PATH,
            valuation_pdf_path=VALUATION_PDF_PATH
        )

        # Run analysis
        report = orchestrator.run_analysis(COMPANY_NAME)

        # Save report
        output_filename = f"investment_report_{COMPANY_NAME.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_file = os.path.join(OUTPUT_PATH, output_filename)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\n✓ Report saved to: {output_file}")
        print(f"\nTo view the report:")
        print(f"  cat {output_file}")
        print()

        # Also print to console
        print("\n" + "="*80)
        print("REPORT PREVIEW")
        print("="*80)
        print(report)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED")
        print(f"{'='*80}")
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()