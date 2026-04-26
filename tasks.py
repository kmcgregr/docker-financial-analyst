"""
Financial Analysis Tasks
Defines all tasks for the financial analysis workflow.

Fixes applied vs v2:
  D - valuation_params no longer bulk-injected into Task 4 at creation time.
      create_valuation_task() embeds a {valuation_placeholder} token instead.
      The orchestrator calls rag_query_fn() at runtime — after the growth
      analyst has run — so the RAG query is informed by actual findings.
  E - PipelineStep dataclass introduced. Agent/task binding is now explicit
      and named. zip() silent-misalignment risk eliminated.

Previous fixes retained:
  A - extracted_docs / docs_context removed from Tasks 2-5.
  B - {context_placeholder} injected + capped by orchestrator.
  C - Structured SUMMARY_PROMPT_SUFFIX appended by orchestrator.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain.chains import LLMChain


# ---------------------------------------------------------------------------
# Fix E — Explicit pipeline step binding
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    """
    Binds a named agent to its task explicitly.

    Replaces the implicit zip(agents, tasks) pattern that silently dropped
    steps whenever the two lists fell out of sync.

    The valuation step additionally carries:
      _rag_query_fn    — called at runtime with a targeted query string
      _fallback_params — used if rag_query_fn is None or raises
    """
    name:  str
    agent: LLMChain
    task:  Dict[str, Any]

    # Fix D: attached by create_pipeline() on the valuation step only
    _rag_query_fn:    Optional[Callable[[str], str]] = field(default=None, repr=False)
    _fallback_params: str                             = field(default="",   repr=False)


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------

class FinancialTasks:
    """Factory class for creating financial analysis tasks."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Task 1 — Document Analyst
    # Only task that receives raw PDF content.
    # ------------------------------------------------------------------

    def create_document_extraction_task(
        self,
        extracted_docs: Dict[str, str],
        company_name: str,
    ) -> Dict[str, Any]:
        """Extract and organise all financial data from raw PDF content."""

        docs_context = self._format_documents(extracted_docs)

        return {
            "input": f"""Analyze the following financial documents for {company_name} and extract
all relevant financial data in a comprehensive, organized manner.

Extract and organize:

1. COMPANY INFORMATION
   - Company name and ticker symbol (if available)
   - Fiscal year and reporting periods
   - Industry / sector

2. INCOME STATEMENT DATA
   - Revenue (by quarter/year with growth rates)
   - Cost of revenue / COGS
   - Gross profit and gross margin %
   - Operating expenses (breakdown if available)
   - Operating income and operating margin %
   - Net income and net margin %
   - Earnings per share (EPS)

3. BALANCE SHEET DATA
   - Total assets
   - Current assets (cash, receivables, inventory)
   - Total liabilities
   - Current liabilities
   - Shareholders' equity
   - Key ratios (current ratio, debt-to-equity, etc.)

4. CASH FLOW DATA
   - Operating cash flow
   - Investing cash flow
   - Financing cash flow
   - Free cash flow
   - Capital expenditures

5. KEY METRICS & RATIOS
   - Return on equity (ROE)
   - Return on assets (ROA)
   - Any company-specific KPIs mentioned

Financial Documents:
{docs_context}

Provide a STRUCTURED, DETAILED summary with all numbers clearly labelled with
their reporting periods. If data is missing or unclear, note that explicitly.
Calculate any obvious growth rates or trends.

Expected output:
Comprehensive structured financial data extraction organized by:
- Company information section
- Income statement metrics with trends
- Balance sheet snapshot with key ratios
- Cash flow analysis
- Calculated growth rates and financial health indicators
All data clearly labelled with periods and units.""",
        }

    # ------------------------------------------------------------------
    # Task 2 — Business Model Analyst
    # ------------------------------------------------------------------

    def create_business_analysis_task(
        self,
        company_name: str,
        context: str,
    ) -> Dict[str, Any]:
        """Analyse the company's business model from prior extracted data."""

        return {
            "input": f"""Based on the structured financial data extracted below,
provide a comprehensive analysis of {company_name}'s business model.

Previous analysis context:
{context}

Analyze and explain:

1. BUSINESS OVERVIEW
   - What does the company do? (core products/services)
   - What problem does it solve for customers?
   - Brief company history/background if available

2. REVENUE MODEL
   - How does the company make money?
   - What are the primary revenue streams?
   - Is it B2B, B2C, or both?
   - Revenue model type (subscription, transactional, licensing, etc.)
   - Geographic revenue breakdown if available

3. CUSTOMER BASE
   - Who are the target customers?
   - What customer segments does it serve?
   - Any information on customer concentration or diversity

4. COMPETITIVE POSITIONING
   - What is the company's competitive advantage (moat)?
   - Market position (leader, challenger, niche player?)
   - Any mentioned competitive threats or advantages

5. BUSINESS QUALITY ASSESSMENT
   - Business model sustainability
   - Scalability potential
   - Cyclicality vs. recurring revenue
   - Any regulatory or market risks mentioned

Synthesize information from the extracted data above and provide clear explanations.
If certain information is not available, note that and make reasonable inferences
based on the financial data patterns.

Expected output:
- Plain-language explanation of what the company does
- Detailed revenue model breakdown
- Customer base and market positioning
- Competitive advantages and moats
- Business quality assessment with specific supporting evidence

Do NOT ask for additional documents — work entirely from the extracted data above.""",
        }

    # ------------------------------------------------------------------
    # Task 3 — Growth & Revenue Analyst
    # ------------------------------------------------------------------

    def create_growth_analysis_task(
        self,
        company_name: str,
        context: str,
    ) -> Dict[str, Any]:
        """Analyse growth trajectory and KPIs from prior extracted data."""

        return {
            "input": f"""Conduct a thorough analysis of {company_name}'s growth trajectory,
key performance indicators, and pricing power based on the context below.

Previous analysis context:
{context}

Analyze the following:

1. REVENUE GROWTH ANALYSIS
   - Calculate quarter-over-quarter (QoQ) growth rates
   - Calculate year-over-year (YoY) growth rates
   - Calculate CAGR if multiple years available
   - Identify acceleration or deceleration trends
   - Compare to industry benchmarks if mentioned

2. PROFITABILITY TRENDS
   - Gross margin trends over time
   - Operating margin trends over time
   - Net margin trends over time
   - Are margins expanding or contracting?

3. KEY PERFORMANCE INDICATORS
   - Identify all company-specific KPIs mentioned
   - Analyse KPI trends (customer growth, retention, etc.)
   - Evaluate KPI health and trajectory
   - Industry-standard metrics if applicable

4. PRICING POWER ASSESSMENT
   - Evidence of pricing power in margin trends
   - Revenue growth vs. volume growth indicators
   - Premium vs. commodity pricing position
   - Ability to pass costs to customers

5. GROWTH QUALITY & SUSTAINABILITY
   - Is growth organic or acquisition-driven?
   - Revenue quality (recurring vs. one-time)
   - Cash generation vs. accounting profits
   - Sustainability of current growth rates

Provide specific calculations with percentages and show your work clearly.
Identify both positive momentum and concerning trends.

Expected output:
- Specific growth rate calculations (QoQ, YoY, CAGR) with numbers
- Margin trend analysis with data points
- Complete KPI assessment with trajectories
- Clear pricing power evaluation with evidence
- Growth quality assessment with supporting metrics

Do NOT ask for additional documents — work entirely from the extracted data above.""",
        }

    # ------------------------------------------------------------------
    # Task 4 — Valuation Specialist
    #
    # Fix D: valuation_params is NO LONGER injected at task-creation time.
    #
    # The prompt contains a {valuation_placeholder} token. The orchestrator
    # replaces it just before this step runs by calling:
    #   step._rag_query_fn(targeted_query)
    # where targeted_query is built from the growth analyst's summary block.
    # This means the RAG retrieval is informed by the actual growth metrics
    # found (e.g. "high gross margins, 40% ARR growth, SaaS model") rather
    # than a generic upfront fetch.
    #
    # If rag_query_fn is None or raises, the orchestrator falls back to
    # step._fallback_params (the pre-fetched broad valuation_params string).
    # ------------------------------------------------------------------

    def create_valuation_task(
        self,
        company_name: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Valuation task prompt.

        Runtime placeholders (both injected by the orchestrator):
          {context_placeholder}      — capped prior-agent summaries (Fix B)
          {valuation_placeholder}    — targeted RAG result (Fix D)
        """

        return {
            "input": f"""Perform a comprehensive valuation analysis of {company_name}
using the valuation parameters retrieved below and the prior analysis context.

Previous analysis context:
{context}

Your analysis should include:

1. MULTIPLE-BASED VALUATION
   - Calculate relevant multiples (P/E, P/S, EV/EBITDA, P/B, etc.)
   - Compare to industry averages/ranges from the parameters below
   - Determine if multiples suggest overvaluation or undervaluation

2. INTRINSIC VALUE CALCULATION
   - Apply DCF or other intrinsic value methods from the parameters below
   - Use appropriate discount rates from the parameters below
   - Make reasonable growth assumptions based on historical data in context
   - Calculate fair value estimate

3. COMPARABLE ANALYSIS
   - Compare to peer companies if benchmarks are provided
   - Adjust for size, growth, and profitability differences
   - Determine relative valuation

4. VALUATION RANGE
   - Synthesize multiple approaches
   - Provide bear, base, and bull case valuations
   - Explain key assumptions and sensitivities

5. VALUATION OPINION
   - Is the stock overvalued, fairly valued, or undervalued?
   - What is the implied upside/downside?
   - Key value drivers and risks to valuation

Valuation Parameters and Methodologies:
{{valuation_placeholder}}

Show all calculations clearly. Use the methodologies and parameters provided.
Be explicit about assumptions and their impact on valuation.

Expected output:
- Multiple valuation approaches with calculations shown
- Clear fair value estimate or range
- Comparison to current price (if available)
- Bear / base / bull scenarios
- Key assumptions and sensitivities clearly stated
- Final valuation opinion (overvalued / fairly valued / undervalued)

Do NOT ask for additional documents — work entirely from the context and
valuation parameters provided above.""",
        }

    # ------------------------------------------------------------------
    # Task 5 — Investment Advisor
    # ------------------------------------------------------------------

    def create_investment_recommendation_task(
        self,
        company_name: str,
        context: str,
    ) -> Dict[str, Any]:
        """Synthesise all prior analyses into a final investment recommendation."""

        return {
            "input": f"""As a Senior Investment Advisor, synthesize all previous
analyses to provide a clear, actionable investment recommendation for {company_name}.

Previous analysis context:
{context}

Your recommendation must include:

1. INVESTMENT RATING
   - Clear rating: STRONG BUY, BUY, HOLD, SELL, or STRONG SELL
   - Conviction level: High, Medium, or Low

2. INVESTMENT THESIS (3-5 key points)
   - Why this is a good/bad investment opportunity
   - Key strengths supporting the thesis
   - What makes this company attractive or unattractive

3. SUPPORTING EVIDENCE
   - Financial health summary
   - Business model strength
   - Growth prospects and sustainability
   - Valuation attractiveness
   - Competitive positioning

4. KEY RISKS (Top 3-5)
   - What could go wrong?
   - Execution risks
   - Market / competitive risks
   - Financial risks
   - Regulatory or other external risks

5. VALUATION PERSPECTIVE
   - Current valuation assessment
   - Target price or price range (if applicable)
   - Expected return potential
   - Time horizon considerations

6. WHO SHOULD INVEST
   - Investor profile suited for this stock
   - Risk tolerance needed
   - Investment time horizon

7. FINAL VERDICT
   - One paragraph summary of your recommendation
   - Clear action item for the reader

Be honest, balanced, and specific. Consider both bull and bear cases.
Back up opinions with evidence from the analyses above.

Expected output:
- Explicit rating (Strong Buy / Buy / Hold / Sell / Strong Sell)
- Concise investment thesis (3-5 bullet points)
- Supporting evidence from all analyses
- Top risks clearly identified
- Valuation perspective and target
- Investor suitability assessment
- Final verdict paragraph with clear action
Written in professional yet accessible language.""",
        }

    # ------------------------------------------------------------------
    # Fix E — Pipeline factory: returns PipelineStep list
    # ------------------------------------------------------------------

    def create_pipeline(
        self,
        agents: List[LLMChain],
        extracted_docs: Dict[str, str],
        valuation_params: str,
        company_name: str,
        rag_query_fn: Optional[Callable[[str], str]] = None,
    ) -> List[PipelineStep]:
        """
        Build and return an ordered list of PipelineStep objects.

        Each step explicitly names the agent, its task, and its position in
        the pipeline — eliminating the implicit zip() coupling.

        Args:
            agents:          Ordered list from FinancialAgents.create_agents().
                             Must have exactly 5 elements.
            extracted_docs:  {filename: content} from the extraction step.
            valuation_params: Broad fallback valuation text (used if
                             rag_query_fn is None or raises at runtime).
            company_name:    Company being analysed.
            rag_query_fn:    Optional callable — signature (query: str) -> str.
                             Called just before the Valuation step runs so the
                             RAG query can be targeted to growth findings.
        """
        if len(agents) != 5:
            raise ValueError(
                f"Expected exactly 5 agents, got {len(agents)}. "
                "Ensure FinancialAgents.create_agents() returns all 5 agents."
            )

        steps: List[PipelineStep] = [
            PipelineStep(
                name="Document Analyst",
                agent=agents[0],
                task=self.create_document_extraction_task(
                    extracted_docs=extracted_docs,
                    company_name=company_name,
                ),
            ),
            PipelineStep(
                name="Business Analyst",
                agent=agents[1],
                task=self.create_business_analysis_task(
                    company_name=company_name,
                    context="{context_placeholder}",
                ),
            ),
            PipelineStep(
                name="Growth Analyst",
                agent=agents[2],
                task=self.create_growth_analysis_task(
                    company_name=company_name,
                    context="{context_placeholder}",
                ),
            ),
            PipelineStep(
                name="Valuation Specialist",
                agent=agents[3],
                task=self.create_valuation_task(
                    company_name=company_name,
                    context="{context_placeholder}",
                ),
                # Fix D: RAG query fn + fallback attached here.
                # Orchestrator calls _rag_query_fn just before this step runs.
                _rag_query_fn=rag_query_fn,
                _fallback_params=valuation_params,
            ),
            PipelineStep(
                name="Investment Advisor",
                agent=agents[4],
                task=self.create_investment_recommendation_task(
                    company_name=company_name,
                    context="{context_placeholder}",
                ),
            ),
        ]

        return steps

    # ------------------------------------------------------------------
    # Backward-compat shim
    # ------------------------------------------------------------------

    def create_tasks(
        self,
        extracted_docs: Dict[str, str],
        valuation_params: str,
        company_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Deprecated — use create_pipeline() instead.
        Retained so any existing callers don't break immediately.
        """
        warnings.warn(
            "create_tasks() is deprecated. Switch to create_pipeline() which "
            "returns PipelineStep objects with explicit agent/task bindings "
            "and runtime RAG injection (Fix D/E).",
            DeprecationWarning,
            stacklevel=2,
        )
        return [
            self.create_document_extraction_task(extracted_docs, company_name),
            self.create_business_analysis_task(company_name, "{context_placeholder}"),
            self.create_growth_analysis_task(company_name, "{context_placeholder}"),
            self.create_valuation_task(company_name, "{context_placeholder}"),
            self.create_investment_recommendation_task(company_name, "{context_placeholder}"),
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _format_documents(extracted_docs: Dict[str, str]) -> str:
        """Format extracted documents for inclusion in Task 1's prompt."""
        lines = []
        for filename, content in extracted_docs.items():
            lines.append(f"\n{'='*70}")
            lines.append(f"DOCUMENT: {filename}")
            lines.append("=" * 70)
            lines.append(content)
        return "\n".join(lines)